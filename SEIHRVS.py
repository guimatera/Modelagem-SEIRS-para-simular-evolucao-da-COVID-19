import numpy as np

# Mapeamento dos Índices no Vetor de Estado Etário
# Grupos Etários: 0 -> 0-4a, 1 -> 5-29a, 2 -> 30-59a (60a+ alocados em R na condição inicial)
IDX_S  = (0, 1, 2)
IDX_S1 = (3, 4, 5)
IDX_V  = (6, 7, 8)

# Compartimentos Clínicos e Desfechos
IDX_E     = 9
IDX_I     = 10
IDX_H     = 11
IDX_C     = 12
IDX_R     = 13
IDX_D_NAT = 14
IDX_D_DIS = 15

N_VIVOS = 14
N_COMPARTIMENTOS = 16

# Frações de 0-4, 5-29 e 30-59 anos em 2016 (projeção IBGE, edição 2024); o restante é 60+
FRACOES_POPULACAO = (14705019 / 203871925, 80916904 / 203871925, 82378614 / 203871925)

# Importação de casos, cenário 2007-2026 de parametros/estimar_importacao.py
MU_IMPORTACAO = 2.532  # eventos/ano
TAMANHOS_IMPORTACAO = [1] * 21 + [2] * 7 + [3] * 3 + [4] * 7 + [5, 6, 6, 9, 14, 18, 18, 24, 33, 40, 41, 133]


def sobrevivencia_media(pi_taxa, tau):
    # A(tau): integral de exp(-pi*a) para a de 0 até tau
    if pi_taxa == 0:
        return tau
    return (1.0 - np.exp(-pi_taxa * tau)) / pi_taxa


def beta_de_R0(R0, params):
    # R0 = beta * exp(-pi*tau_E) * [pH*A(tau_IH) + (1-pH)*A(tau_IR)]
    pi_taxa = params["Pi"]
    permanencia_I = (params["pH"] * sobrevivencia_media(pi_taxa, params["tau_IH"]) +
                     (1.0 - params["pH"]) * sobrevivencia_media(pi_taxa, params["tau_IR"]))
    return R0 / (np.exp(-pi_taxa * params["tau_E"]) * permanencia_I)


def sortear_epsilon(mu_ano, tamanhos, nt, dt, rng):
    # Eventos de importação: a cada ano, número de eventos ~ Poisson(mu_ano);
    # dia uniforme no ano e n_k sorteado de tamanhos
    eps = np.zeros(nt)
    passos_ano = int(round(365 / dt))
    for inicio in range(0, nt, passos_ano):
        passos = min(passos_ano, nt - inicio)
        n_eventos = rng.poisson(mu_ano * passos / passos_ano)
        if n_eventos > 0:
            dias = rng.integers(inicio, inicio + passos, n_eventos)
            np.add.at(eps, dias, rng.choice(tamanhos, n_eventos) / dt)
    return eps


def faixas_de_N(N):
    # (0-4a, 5-29a, 30-59a, 60a+) a partir do total, com as frações de 2016
    N_g1, N_g2, N_g3 = (int(N * f) for f in FRACOES_POPULACAO)
    return N_g1, N_g2, N_g3, N - (N_g1 + N_g2 + N_g3)


def condicao_inicial(faixas, E0, I0, H0, C0):
    N_g1, N_g2, N_g3, N_60 = faixas

    x0 = np.zeros(N_COMPARTIMENTOS)
    x0[IDX_E], x0[IDX_I], x0[IDX_H], x0[IDX_C] = E0, I0, H0, C0
    x0[IDX_S[0]] = N_g1 - (E0 + I0 + H0 + C0)
    x0[IDX_S[1]] = N_g2
    x0[IDX_S[2]] = N_g3
    x0[IDX_R] = N_60
    return x0


def simular(x0, t0, tf, dt, params, epsilon=None, fluxos=None):
    t = np.arange(t0, tf, dt)
    nt = t.size
    pi_taxa, Lambda = params["Pi"], params["Lambda"]
    pH, pC, d = params["pH"], params["pC"], params["d"]
    e1, e2 = params["e1"], params["e2"]
    c1, c2 = params["c1"], (params["c2_1"], params["c2_2"])
    v1 = (params["v1_2"], params["v1_3"])
    tau_v2 = (params["tau_v2_1"], params["tau_v2_2"])
    tau_E, tau_IH, tau_IR = params["tau_E"], params["tau_IH"], params["tau_IR"]
    tau_HC, tau_HR, tau_HD = params["tau_HC"], params["tau_HR"], params["tau_HD"]
    tau_CR, tau_CD, tau_D1 = params["tau_CR"], params["tau_CD"], params["tau_D1"]
    beta = beta_de_R0(params["R0"], params)
    icu_10mil = params["ICU"]

    if epsilon is None:
        epsilon = np.zeros(nt)

    def A(tau):
        return sobrevivencia_media(pi_taxa, tau)

    def S_exp(tau):
        return np.exp(-pi_taxa * tau)

    # História inicial: entradas constantes para t < t0, coerentes com os estoques em t0
    pre_E = x0[IDX_E] / A(tau_E)
    pre_I = x0[IDX_I] / (pH * A(tau_IH) + (1.0 - pH) * A(tau_IR))
    pre_H = x0[IDX_H] / ((1.0 - pC) * A(tau_HR) + pC * A(tau_HC))
    pre_C = x0[IDX_C] / ((1.0 - d) * A(tau_CR) + d * A(tau_CD))

    # Séries das entradas de cada passo, usadas pelos fluxos atrasados
    ent_E = np.zeros(nt)
    ent_I = np.zeros(nt)
    ent_H = np.zeros(nt)
    ent_C = np.zeros(nt)
    sem_leito = np.zeros(nt)
    falha_g1 = np.zeros(nt)
    falha_g2 = np.zeros(nt)
    risco_acum = np.zeros(nt)  # integral de (pi + lambda)

    def atrasado(serie, k, tau, pre):
        # Média da entrada constante por passo na janela [t_k - tau, t_k - tau + dt]:
        # toda a massa que entra sai depois do atraso, mesmo com tau fracionário
        pos = min(k - tau / dt, k - 1.0)
        i = int(np.floor(pos))
        frac = pos - i
        anterior = pre if i < 0 else serie[i]
        seguinte = pre if i + 1 < 0 else serie[i + 1]
        return (1.0 - frac) * anterior + frac * seguinte

    def sobrevive_infeccao(k, tau):
        # exp(-integral de (pi + lambda) em [t - tau, t]); antes de t0, lambda = 0
        pos = k - tau / dt
        if pos < 0:
            passado = pi_taxa * pos * dt
        else:
            i = int(np.floor(pos))
            frac = pos - i
            seguinte = risco_acum[min(i + 1, k)]
            passado = (1.0 - frac) * risco_acum[i] + frac * seguinte
        return np.exp(-(risco_acum[k] - passado))

    def fluxos_atrasados(k, x_k, icu_k):
        # Fluxos que dependem só do passado: constantes dentro do passo
        f = {}
        # FLUXOS VACINAIS PNI BRASIL
        # Grupo 1 (0-4a): coorte que completa 12 meses
        f["d1_g1"] = c1 * Lambda * sobrevive_infeccao(k, tau_D1)
        f["d2_g1"] = e2 * c2[0] * atrasado(falha_g1, k, tau_v2[0], 0.0) * sobrevive_infeccao(k, tau_v2[0])
        # Grupo 2 (5-29a)
        f["d2_g2"] = e2 * c2[1] * atrasado(falha_g2, k, tau_v2[1], 0.0) * sobrevive_infeccao(k, tau_v2[1])

        # Progressão Clínica
        f["E_I"] = atrasado(ent_E, k, tau_E, pre_E) * S_exp(tau_E)
        f["I_H"] = pH * atrasado(ent_I, k, tau_IH, pre_I) * S_exp(tau_IH)
        f["I_R"] = (1.0 - pH) * atrasado(ent_I, k, tau_IR, pre_I) * S_exp(tau_IR)

        # Admissão limitada aos leitos livres: quem excede a capacidade fica sem leito
        potencial = pC * atrasado(ent_H, k, tau_HC, pre_H) * S_exp(tau_HC)
        livres = max(icu_k - x_k[IDX_C], 0.0)
        f["H_C"] = min(potencial, livres / dt)
        f["sem_leito"] = potencial - f["H_C"]
        f["H_D"] = atrasado(sem_leito, k, tau_HD, 0.0) * S_exp(tau_HD)
        f["H_R"] = (1.0 - pC) * atrasado(ent_H, k, tau_HR, pre_H) * S_exp(tau_HR)

        f["C_R"] = (1.0 - d) * atrasado(ent_C, k, tau_CR, pre_C) * S_exp(tau_CR)
        f["C_D"] = d * atrasado(ent_C, k, tau_CD, pre_C) * S_exp(tau_CD)
        return f

    def derivadas(z, t_atual, f, eps_k, fim_semente):
        x = z[:N_COMPARTIMENTOS]
        I = x[IDX_I]
        N_t = x[:N_VIVOS].sum()

        # Corte de transmissão: I < 1 não transmite, salvo na janela infecciosa de uma semente
        ativo = I >= 1.0 or t_atual <= fim_semente
        lam = beta * I / N_t if ativo else 0.0

        S, S1, V = x[list(IDX_S)], x[list(IDX_S1)], x[list(IDX_V)]
        d1_g2 = v1[0] * S[1]
        d1_g3 = v1[1] * S[2]
        ent_E_t = lam * (S.sum() + S1.sum())

        dz = np.zeros(z.size)
        # Grupo 1
        dz[IDX_S[0]]  = Lambda - lam * S[0] - f["d1_g1"] - pi_taxa * S[0]
        dz[IDX_S1[0]] = (1.0 - e1) * f["d1_g1"] - lam * S1[0] - f["d2_g1"] - pi_taxa * S1[0]
        dz[IDX_V[0]]  = e1 * f["d1_g1"] + f["d2_g1"] - pi_taxa * V[0]
        # Grupo 2
        dz[IDX_S[1]]  = - lam * S[1] - d1_g2 - pi_taxa * S[1]
        dz[IDX_S1[1]] = (1.0 - e1) * d1_g2 - lam * S1[1] - f["d2_g2"] - pi_taxa * S1[1]
        dz[IDX_V[1]]  = e1 * d1_g2 + f["d2_g2"] - pi_taxa * V[1]
        # Grupo 3 (Dose Única)
        dz[IDX_S[2]]  = - lam * S[2] - d1_g3 - pi_taxa * S[2]
        dz[IDX_S1[2]] = (1.0 - e1) * d1_g3 - lam * S1[2] - pi_taxa * S1[2]  # Sem D2
        dz[IDX_V[2]]  = e1 * d1_g3 - pi_taxa * V[2]

        # Progressão Clínica
        dz[IDX_E] = ent_E_t - f["E_I"] - pi_taxa * x[IDX_E]
        dz[IDX_I] = f["E_I"] + eps_k - f["I_H"] - f["I_R"] - pi_taxa * I
        dz[IDX_H] = f["I_H"] - f["H_C"] - f["H_D"] - f["H_R"] - pi_taxa * x[IDX_H]
        dz[IDX_C] = f["H_C"] - f["C_R"] - f["C_D"] - pi_taxa * x[IDX_C]
        dz[IDX_R] = f["I_R"] + f["H_R"] + f["C_R"] - pi_taxa * x[IDX_R]

        dz[IDX_D_NAT] = pi_taxa * N_t
        dz[IDX_D_DIS] = f["H_D"] + f["C_D"]

        # Integrais do passo, gravadas nas séries depois do RK4
        dz[N_COMPARTIMENTOS] = ent_E_t
        dz[N_COMPARTIMENTOS + 1] = (1.0 - e1) * d1_g2
        dz[N_COMPARTIMENTOS + 2] = lam
        return dz

    x = np.zeros([N_COMPARTIMENTOS, nt])
    x[:, 0] = x0
    rt = np.zeros(nt)
    icu = np.zeros(nt)

    # Sementes iniciais em E ou I transmitem até o fim da primeira janela infecciosa
    fim_semente = (t0 + tau_E + tau_IR) if (x0[IDX_E] + x0[IDX_I]) > 0 else t0 - 1.0

    for k in range(nt):
        xk = x[:, k]
        N_t = xk[:N_VIVOS].sum()
        icu[k] = icu_10mil / 10000 * N_t

        S_tot = xk[list(IDX_S)].sum() + xk[list(IDX_S1)].sum()
        rt[k] = params["R0"] * S_tot / N_t

        if k == nt - 1:
            break

        if epsilon[k] > 0:
            fim_semente = max(fim_semente, t[k] + tau_IR)

        f = fluxos_atrasados(k, xk, icu[k])
        ent_I[k] = f["E_I"] + epsilon[k]
        ent_H[k] = f["I_H"]
        ent_C[k] = f["H_C"]
        sem_leito[k] = f["sem_leito"]
        falha_g1[k] = (1.0 - e1) * f["d1_g1"]

        # RK4: fluxos atrasados e epsilon ficam constantes dentro do passo
        args = (f, epsilon[k], fim_semente)
        z = np.concatenate([xk, np.zeros(3)])
        k1 = dt * derivadas(z, t[k], *args)
        k2 = dt * derivadas(z + k1 / 2, t[k] + dt / 2, *args)
        k3 = dt * derivadas(z + k2 / 2, t[k] + dt / 2, *args)
        k4 = dt * derivadas(z + k3, t[k] + dt, *args)
        z_novo = z + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        x[:, k + 1] = z_novo[:N_COMPARTIMENTOS]
        ent_E[k] = z_novo[N_COMPARTIMENTOS] / dt
        falha_g2[k] = z_novo[N_COMPARTIMENTOS + 1] / dt
        risco_acum[k + 1] = risco_acum[k] + dt * pi_taxa + z_novo[N_COMPARTIMENTOS + 2]

    # Entradas por dia em cada compartimento, para comparar com séries de eventos
    if fluxos is not None:
        fluxos.update(ent_I=ent_I, ent_H=ent_H, ent_C=ent_C)

    return x, t, rt, icu


def main():
    import PySimpleGUI as sg
    import matplotlib.pyplot as plt

    # Tema da interface gráfica
    sg.theme('Default1')

    font_size = 8
    font_style = 'Helvetica'
    fonte = (font_style, font_size)
    fonte_titulo = (font_style, font_size + 2)
    sg.set_options(font=fonte)

    # Layout da Interface Gráfica
    left_col = [
        [sg.Text('Parâmetros Gerais e Demográficos:', font=fonte_titulo)],
        [sg.Text('Tamanho da população (N):')],
        [sg.Slider(range=(0, 8e9), default_value=213421037, resolution=10000, size=(45, 10), orientation='horizontal', key='-popsize-')],
        [sg.Text('Nascimentos (Λ diário):')],
        [sg.Slider(range=(0, 2e5), default_value=1000, resolution=1, size=(45, 10), orientation='horizontal', key='-nascimentos-')],
        [sg.Text('R0 (Número de Reprodução Básica):')],
        [sg.Slider(range=(0, 25), default_value=12, resolution=0.1, size=(45, 10), orientation='horizontal', key='-repr-')],
        [sg.Text('Tempo de Simulação (anos):')],
        [sg.Slider(range=(0, 20), default_value=5, resolution=1, size=(45, 10), orientation='horizontal', key='-time-')],
        [sg.Text('Taxa de mortalidade natural (dias - π): '), sg.Slider(range=(0, 1e-4), default_value=2e-5, resolution=1e-5, size=(10, 10), orientation='horizontal', key='-mortalidadeNatural-')],
        [sg.Text('Importação de Casos (ε):', font=fonte_titulo)],
        [sg.Text('Frequência de eventos μ (eventos/ano):')],
        [sg.Slider(range=(0, 20), default_value=MU_IMPORTACAO, resolution=0.001, size=(45, 10), orientation='horizontal', key='-mu-')],
        [sg.Text('Casos por evento (n_k): sorteados dos eventos do SINAN')],
    ]

    middle_col = [
        [sg.Text('Regras de Vacinação (PNI / MS Brasil):', font=fonte_titulo)],
        [sg.Text('Vacinação Ativa: '), sg.Spin(values=('No', 'Yes'), initial_value='Yes', size=(5, 10), key='-vacinacao-')],
        [sg.Text('Eficácia 1ª Dose (e1): '), sg.Slider(range=(0, 1), default_value=0.93, resolution=0.01, size=(30, 10), orientation='horizontal', key='-efetividade1-')],
        [sg.Text('Eficácia 2ª Dose nas falhas (e2): '), sg.Slider(range=(0, 1), default_value=0.571, resolution=0.001, size=(30, 10), orientation='horizontal', key='-efetividade2-')],
        [sg.Text('Cobertura D1 aos 12 meses (c1): '), sg.Slider(range=(0, 1), default_value=1, resolution=0.01, size=(30, 10), orientation='horizontal', key='-c1-')],
        [sg.Text('Cobertura D2 Infantil (c2,1): '), sg.Slider(range=(0, 1), default_value=1, resolution=0.01, size=(30, 10), orientation='horizontal', key='-c2_1-')],
        [sg.Text('Cobertura D2 5-29 anos (c2,2): '), sg.Slider(range=(0, 1), default_value=1, resolution=0.01, size=(30, 10), orientation='horizontal', key='-c2_2-')],
        [sg.Text('Intervalo D1->D2 Infantil (90 dias): '), sg.Text('Fixo aos 15m (Tetraviral)')],
        [sg.Text('Intervalo D1->D2 5-29 anos (30 dias): '), sg.Text('Fixo (Catch-up / Surtos)')],
        [sg.Text('Faixa 30-59 anos: '), sg.Text('Dose Única (Sem D2)')],

        [sg.Text('Dinâmica Hospitalar:', font=fonte_titulo)],
        [sg.Text('UTIs disponíveis (por 10 mil hab):')],
        [sg.Slider(range=(0, 20), default_value=2.43, resolution=0.01, size=(45, 10), orientation='horizontal', key='-uti-')],
        [sg.Text('Fração de I hospitalizada (p_H):')],
        [sg.Slider(range=(0, 1), default_value=0.2, resolution=0.001, size=(45, 10), orientation='horizontal', key='-pH-')],
        [sg.Text('Fração para UTI (p_C):')],
        [sg.Slider(range=(0, 1), default_value=0.032, resolution=0.001, size=(45, 10), orientation='horizontal', key='-pC-')],
        [sg.Text('Letalidade UTI (d):')],
        [sg.Slider(range=(0, 1), default_value=0.0804, resolution=0.0001, size=(45, 10), orientation='horizontal', key='-letalidadeUTI-')],
    ]

    atrasos_col = [
        [sg.Text('Atrasos Clínicos (dias):', font=fonte_titulo)],
        [sg.Text('τ_E (E→I): '), sg.Slider(range=(1, 30), default_value=12.5, resolution=0.01, size=(20, 10), orientation='horizontal', key='-tauE-')],
        [sg.Text('τ_IH (I→H, SINAN): '), sg.Slider(range=(1, 30), default_value=4.23, resolution=0.01, size=(20, 10), orientation='horizontal', key='-tauIH-')],
        [sg.Text('τ_IR (I→R): '), sg.Slider(range=(1, 30), default_value=8, resolution=0.01, size=(20, 10), orientation='horizontal', key='-tauIR-')],
        [sg.Text('τ_HC (H→C): '), sg.Slider(range=(1, 30), default_value=6.7, resolution=0.01, size=(20, 10), orientation='horizontal', key='-tauHC-')],
        [sg.Text('τ_HR (H→R): '), sg.Slider(range=(1, 30), default_value=4.47, resolution=0.01, size=(20, 10), orientation='horizontal', key='-tauHR-')],
        [sg.Text('τ_HD (H→D): '), sg.Slider(range=(1, 30), default_value=4.8, resolution=0.01, size=(20, 10), orientation='horizontal', key='-tauHD-')],
        [sg.Text('τ_CR (C→R): '), sg.Slider(range=(1, 60), default_value=14, resolution=0.01, size=(20, 10), orientation='horizontal', key='-tauCR-')],
        [sg.Text('τ_CD (C→D): '), sg.Slider(range=(1, 60), default_value=7.9, resolution=0.01, size=(20, 10), orientation='horizontal', key='-tauCD-')],
    ]

    right_col = [
        [sg.Text('Condições Iniciais:', font=fonte_titulo)],
        [sg.Text('Expostos (E):')],
        [sg.Slider(range=(0, 2e6), default_value=1, resolution=1, size=(45, 10), orientation='horizontal', key='-E_0-')],
        [sg.Text('Infecciosos (I):')],
        [sg.Slider(range=(0, 2e6), default_value=0, resolution=1, size=(45, 10), orientation='horizontal', key='-I_0-')],
        [sg.Text('Hospitalizados Enfermaria (H):')],
        [sg.Slider(range=(0, 2e6), default_value=0, resolution=1, size=(45, 10), orientation='horizontal', key='-H0-')],
        [sg.Text('Pacientes em UTI (C):')],
        [sg.Slider(range=(0, 2e6), default_value=0, resolution=1, size=(45, 10), orientation='horizontal', key='-C0-')],
    ]

    corpo = [
        [
            sg.Column(left_col, vertical_alignment='top'),
            sg.Column(middle_col, vertical_alignment='top'),
            sg.Column(atrasos_col, vertical_alignment='top'),
            sg.Column(right_col, vertical_alignment='top'),
        ]
    ]

    largura_tela, altura_tela = sg.Window.get_screen_size()
    tamanho_corpo = (min(1500, int(largura_tela*0.95)), min(900, int(altura_tela*0.80)))

    layout = [
        [sg.Column(corpo, key='-corpo-', size=tamanho_corpo, scrollable=True, vertical_scroll_only=True)],
        [sg.Button('Ok'), sg.Button('Cancel')]
    ]

    window = sg.Window('Modelo SEIHRVS - DDE Etário (MS Brasil / PNI)', layout, resizable=True, finalize=True)
    window['-corpo-'].expand(True, True)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        if event == 'Ok':
            vacinacao_ativa = values['-vacinacao-'] == 'Yes'

            N = int(values['-popsize-'])
            t0 = 0
            tf = 365 * int(values['-time-'])
            dt = 1.0

            params = {
                'R0': float(values['-repr-']),
                'Pi': float(values['-mortalidadeNatural-']),
                'Lambda': float(values['-nascimentos-']),
                'tau_E': float(values['-tauE-']),
                'tau_IH': float(values['-tauIH-']),
                'tau_IR': float(values['-tauIR-']),
                'tau_HC': float(values['-tauHC-']),
                'tau_HR': float(values['-tauHR-']),
                'tau_HD': float(values['-tauHD-']),
                'tau_CR': float(values['-tauCR-']),
                'tau_CD': float(values['-tauCD-']),
                'pH': float(values['-pH-']),
                'pC': float(values['-pC-']),
                'd': float(values['-letalidadeUTI-']),
                'tau_D1': 365.0,
                'tau_v2_1': 90.0,
                'tau_v2_2': 30.0,
                'e1': float(values['-efetividade1-']),
                'e2': float(values['-efetividade2-']),
                'c1': float(values['-c1-']) if vacinacao_ativa else 0.0,
                'c2_1': float(values['-c2_1-']) if vacinacao_ativa else 0.0,
                'c2_2': float(values['-c2_2-']) if vacinacao_ativa else 0.0,
                'v1_2': 0.002 if vacinacao_ativa else 0.0,
                'v1_3': 0.001 if vacinacao_ativa else 0.0,
                'ICU': float(values['-uti-']),
            }

            x0 = condicao_inicial(faixas_de_N(N), float(values['-E_0-']), float(values['-I_0-']),
                                  float(values['-H0-']), float(values['-C0-']))

            rng = np.random.default_rng()
            nt = np.arange(t0, tf, dt).size
            epsilon = sortear_epsilon(float(values['-mu-']), TAMANHOS_IMPORTACAO, nt, dt, rng)

            x, t, rt, icu = simular(x0, t0, tf, dt, params, epsilon)

            print("RELATÓRIO DE SIMULAÇÃO (DDE ETÁRIO - MS BRASIL / PNI)")
            print("==================================================")
            print("Vacinação PNI Ativa: ", vacinacao_ativa)
            print(f"UTIs Disponíveis: {int(icu[0])}")
            print('Casos importados: ', int(epsilon.sum() * dt))
            print("==================================================")
            print('Óbitos totais: ', int(x[IDX_D_NAT, -1] + x[IDX_D_DIS, -1]))
            print('Óbitos por sarampo: ', int(x[IDX_D_DIS, -1]))

            V_tot_curve = x[list(IDX_V), :].sum(axis=0)
            print('População Imunizada Total (V): ', int(V_tot_curve[-1]))
            print('Pico em UTI (C): ', int(max(x[IDX_C, :])))
            print("==================================================")

            fig, ax = plt.subplots(2, 1, figsize=(10, 8))

            S_tot_curve = x[list(IDX_S), :].sum(axis=0) + x[list(IDX_S1), :].sum(axis=0)

            ax[0].set_title('Simulação Epidemiológica DDE - Faixas Etárias e PNI Brasil')
            ax[0].plot(t/365, S_tot_curve, 'red', label='Suscetíveis Totais (S + S1)')
            ax[0].plot(t/365, V_tot_curve, 'cyan', label='V (Vacinados D1/D2)')
            ax[0].plot(t/365, x[IDX_E, :], 'green', label='E (Expostos)')
            ax[0].plot(t/365, x[IDX_I, :], 'blue', label='I (Infecciosos)')
            ax[0].plot(t/365, x[IDX_H, :], 'magenta', label='H (Enfermaria)')
            ax[0].plot(t/365, x[IDX_C, :], 'darkred', linewidth=1.5, label='C (UTI)')
            ax[0].plot(t/365, icu, linestyle='--', color='black', label='Capacidade de UTI')
            ax[0].set_xlabel('Tempo (anos)')
            ax[0].set_ylabel('População')
            ax[0].grid(True)
            ax[0].legend(loc='upper right', fontsize=7)

            ax[1].set_title('Número de Reprodução Efetivo (Rt)')
            ax[1].plot(t/365, rt, color='purple', label='Rt')
            ax[1].axhline(1.0, color='gray', linestyle=':', label='Limiar epidêmico (Rt = 1)')
            ax[1].set_xlabel('Tempo (anos)')
            ax[1].set_ylabel('Rt')
            ax[1].grid(True)
            ax[1].legend(loc='upper right', fontsize=7)

            plt.subplots_adjust(hspace=0.4)
            plt.show()

    window.close()


if __name__ == '__main__':
    main()
