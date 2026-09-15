# Importando bibliotecas necessárias para o projeto.
import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode, solve_ivp
from scipy.optimize import brentq

# Definindo tema da interface gráfica.
sg.theme('Default1')  

# Criando layout da interface gráfica.   
font_size = 6
font_style = 'Helvetica'
left_col = [ 
            [sg.Text('Parâmetros Gerais:',font=(font_style, font_size+2))], 
            [sg.Text('Tamanho da população:', font=(font_style, font_size))],
            [sg.Slider(range=(0,8e9), default_value=213421037, resolution=10000,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-popsize-')],
            [sg.Text('Nascimentos', font=(font_style, font_size))],
            [sg.Slider(range=(0,2e5), default_value=1000, resolution=1,
            size=(50,10), orientation='horizontal', font=(font_style, font_size), key= '-nascimentos-')], 
            [sg.Text('Imigração:', font=(font_style, font_size))],
            [sg.Slider(range=(-2e5,2e5), default_value=100, resolution=1,
            size=(50,10), orientation='horizontal', font=(font_style, font_size), key= '-imigracao-')], 
            [sg.Text('Quantidade de UTIs disponíveis (a cada 10000 pessoas):', font=(font_style, font_size))],
            [sg.Slider(range=(0,20), default_value=1, resolution=0.1,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-uti-')],
            [sg.Text('Taxa de internação nas UTIs:', font=(font_style, font_size))],
            [sg.Slider(range=(0,1), default_value=0.05, resolution=0.01,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-internacao-')],
            [sg.Text('R0 (Número de Reprodução Básica):', font=(font_style, font_size))],
            [sg.Slider(range=(0,20), default_value=12, resolution=0.1,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-repr-')],
            [sg.Text('Tempo(anos):', font=(font_style, font_size))],
            [sg.Slider(range=(0,10), default_value=5, resolution=1,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-time-')],
            [sg.Text('Nível de distanciamento social:', font=(font_style, font_size))],
            [sg.Slider(range=(0,1), default_value=0.2, resolution=0.01,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-distance-')],
            [sg.Text('Período de Incubação(dias): ', font=(font_style, font_size)), sg.Slider(range=(0,10), default_value=5.1, resolution=0.1,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-incubacao-')], 
            [sg.Text('Período de infecção(dias): ', font=(font_style, font_size)), sg.Slider(range=(0,10), default_value=3.3, resolution=0.1,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-infeccao-')],
            [sg.Text('Período de imunidade(anos) - Recuperados: ', font=(font_style, font_size)), sg.Slider(range=(0,20), default_value=1, resolution=0.5,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-imunidadeNatural-')],
            [sg.Text('Período de imunidade(anos) -  Vacinados: ', font=(font_style, font_size)), sg.Slider(range=(0,20), default_value=5, resolution=0.5,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-imunidadeVacinados-')],
            [sg.Text('Taxa de mortalidade natural(dias): ', font=(font_style, font_size)), sg.Slider(range=(0,1e-4), default_value=2e-5, resolution=1e-5,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-mortalidadeNatural-')],
            [sg.Text('Taxa de mortalidade de Infectados(dias): ', font=(font_style, font_size)), sg.Slider(range=(0,1), default_value=0.1, resolution=0.01,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-mortalidadeInfectados-')],
            [sg.Text('Taxa de mortalidade de Hospitalizados(dias): ', font=(font_style, font_size)), sg.Slider(range=(0,1), default_value=0.3, resolution=0.01,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-mortalidadeHospitalizados-')],
            [sg.Text('Taxa de Suscetíveis Externos(dias): ', font=(font_style, font_size)), sg.Slider(range=(0,1), default_value=0.9998, resolution=0.0001,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-suscetíveisExternos-')],
            [sg.Text('Taxa de Expostos Externos(dias): ', font=(font_style, font_size)), sg.Slider(range=(0,1), default_value=0.0001, resolution=0.0001,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-expostosExternos-')],
            [sg.Text('Taxa de Recuperados Externos(dias): ', font=(font_style, font_size)), sg.Slider(range=(0,1), default_value=0.0001, resolution=0.0001,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-recuperadosExternos-')],
]

middle_col = [ 
    [sg.Text('Condições inciais:')],
    [sg.Text('Pessoas Expostas:')],
    [sg.Slider(range=(0,2e6), default_value=1, resolution=1,
    size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-E0-')],
    [sg.Text('Pessoas Infectadas:')],
    [sg.Slider(range=(0,2e6), default_value=0, resolution=1,
    size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-I0-')],
    [sg.Text('Pessoas Hospitalizadas:')],
    [sg.Slider(range=(0,2e6), default_value=0, resolution=1,
    size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-H0-')],
    [sg.Text('Pessoas Vacinadas:')],
    [sg.Slider(range=(0,2e8), default_value=0, resolution=1,
    size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-V0-')],
    [sg.Text('Pessoas Recuperadas:')],
    [sg.Slider(range=(0,2e6), default_value=0, resolution=1,
    size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-R0-')],
]

right_col = [
    [sg.Text('Ações de Controle: ')],
    [sg.Text('Vacinação: '), sg.Spin(values=('No', 'Yes'), initial_value='No',size=(5,10),
    font=(font_style, font_size),key='-vacinacao-')],
    [sg.Text('Taxa de vacinação por dia: '), sg.Slider(range=(0,1), default_value=0.0021, resolution=0.0001,
    size=(10,10), orientation='horizontal', font=(font_style, font_size), key='-taxa-vacinacao-')],
    [sg.Text('Taxa de efetividade da vacinação: '), sg.Slider(range=(0,1), default_value=0.97, resolution=0.01,
    size=(10,10), orientation='horizontal', font=(font_style, font_size), key='-taxa-efetividade-')],
    [sg.Text('Tempo para início da vacinação: ')],
    [sg.Slider(range=(0,10), default_value=2, resolution=1,
    size=(50,10), orientation='horizontal', font=(font_style, font_size), key='-tempo-vacinacao-')],
    [sg.Text('Lockdown de emergência:'), sg.Spin(values=('No', 'Yes'), initial_value='No',size=(5,10),
    font=(font_style, 8),key='-lockdown-')],
    [sg.Text('Duração do lockdown de emergência(dias):')],
    [sg.Slider(range=(0,60), default_value=30, resolution=1,
    size=(50,10), orientation='horizontal', font=(font_style, font_size), key='-duracao-lockdown-')],
    [sg.Text('Taxa de ocupação limite de leitos: ')],
    [sg.Slider(range=(0,1), default_value=0.5, resolution=0.01,
    size=(50,10), orientation='horizontal', font=(font_style, font_size), key='-icu-condicao-')],
]
           
layout = [
    [
        [sg.Text('Método Numérico: ', font=(font_style, font_size))],
        [sg.Combo(['Runge-Kutta', 'Euler', 'STIFF'], default_value='Runge-Kutta', size=(15, 3), key='-numerical-methods-', readonly=True)],
        sg.Column(left_col, vertical_alignment='top'),
        sg.Column(middle_col, vertical_alignment='top'),
        sg.Column(right_col, vertical_alignment='top'),
    ],
    [sg.Button('Ok'), sg.Button('Cancel')]
]

    

# Criando a janela.
window = sg.Window('Modelo SEIHRVS', layout, finalize=True)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Cancel': # Fecha janela se clicar no ícone de fechar ou no botão de cancelar.
        break
    if event == 'Ok':

        # Função que define as EDO's da modelagem SEIHRVS para uma epidemia.
        def SEIHRVS_MODEL(x, t, params, N, u, ICU, sobrecapacidade=None):
            # Coleta de parâmetros
            alpha = params["Alpha"]
            beta = params["Beta"]
            gammaI = params["GammaI"]
            gammaH = params["GammaH"]
            delta = params["Delta"]
            muI = params["MuI"]
            muH = params["MuH"]
            omegaR = params["OmegaR"]
            omegaV = params["OmegaV"]
            pi = params["Pi"]
            sigma = params["Sigma"]
            zetaS = params["ZetaS"]
            zetaE = params["ZetaE"]
            zetaR = params["zetaR"]
            epsillon = params["Epsillon"]
            e = params["e"] if params["VacinaAtiva"] and t >= params["TempoInicioVacinacao"] else 0.0
            v = params["v"] if params["VacinaAtiva"] and t >= params["TempoInicioVacinacao"] else 0.0
            tau = params["tau"]

            # Decisão entre parâmetro tau inicial ou parâmetro tau de Lockdown
            amort = u if u != tau else tau

            vacina_ativa = params["VacinaAtiva"] and t >= params["TempoInicioVacinacao"]
            fluxo_vacinacao = e*v*x[0] - (pi + omegaV)*x[5] if vacina_ativa else 0.0 
            # Quando informado, fixa o estado de sobrecapacidade das UTIs (usado pelos regimes de capacidade das UTIs).
            sobrecapacidade_uti = x[3] > ICU if sobrecapacidade is None else sobrecapacidade

            # Array com Edos do modelo.
            mortes_naturais = pi * (x[0] + x[1] + x[2] + x[3] + x[4] + x[5])
            mortes_doenca = muI*x[2] + muH*x[3] + (delta*x[2] if sobrecapacidade_uti else 0.0)

            SEIHRVSdot = np.array([-(1-amort)*(beta*x[0]*x[2]/N) + omegaR*x[4] - pi*x[0] - fluxo_vacinacao + sigma + zetaS*epsillon, #dS/dt
                            (1-amort)*(beta*x[0]*x[2]/N) - (alpha + pi)*x[1] + zetaE*epsillon, #dE/dt
                            alpha*x[1] - (gammaI + delta + pi + muI)*x[2], #dI/dt
                            delta*x[2] - (gammaH + muH + pi)*x[3] if not sobrecapacidade_uti else -(gammaH + muH + pi)*x[3], #dH/dt
                            gammaI*x[2] + gammaH*x[3] - (omegaR + pi)*x[4] + zetaR*epsillon, #dR/dt
                            fluxo_vacinacao,  #dV/dt
                            mortes_naturais,
                            mortes_doenca
                            ]) 
            return SEIHRVSdot


        # Regimes da capacidade das UTIs, usados pelo método STIFF e pela solução de referência:
        # 'livre' (H < ICU), 'lotado' (H > ICU) e 'deslizante' (H = ICU). No regime deslizante H fica preso
        # na capacidade (regime deslizante de Filippov): só são internados os que as altas liberam e o
        # excedente de internações vai para os óbitos pela doença.
        def campo_regime(f, tt, xx, tau, regime):
            if regime == 'livre':
                return f(tt, xx, tau, False)
            if regime == 'lotado':
                return f(tt, xx, tau, True)
            # Combinação dos dois campos que mantém dH/dt = 0.
            f_livre = f(tt, xx, tau, False)
            f_lotado = f(tt, xx, tau, True)
            den = f_livre[3] - f_lotado[3]
            lam = np.clip(-f_lotado[3]/den, 0.0, 1.0) if den > 0 else 1.0
            return lam*f_livre + (1 - lam)*f_lotado

        # Função de evento de cada regime: negativa enquanto o regime vale, cruza o zero subindo na troca.
        def evento_regime(f, ICU, tt, xx, tau, regime):
            if regime == 'livre':
                return xx[3] - ICU                  # H atinge a capacidade subindo.
            if regime == 'lotado':
                return ICU - xx[3]                  # H volta à capacidade descendo.
            return -f(tt, xx, tau, False)[3]        # Internações ficam abaixo das altas.

        # Regime quando H está exatamente na capacidade.
        def regime_na_capacidade(f, tt, xx, tau):
            return 'deslizante' if f(tt, xx, tau, False)[3] > 0 else 'livre'

        def regime_inicial(f, ICU, tt, xx, tau):
            if xx[3] < ICU:
                return 'livre'
            if xx[3] > ICU:
                return 'lotado'
            return regime_na_capacidade(f, tt, xx, tau)

        # Aplica a troca de regime no instante do evento (altera y no lugar) e devolve o novo regime.
        def trocar_regime(f, ICU, tt, y, tau, regime):
            y[3] = ICU
            novo = 'livre' if regime == 'deslizante' else regime_na_capacidade(f, tt, y, tau)
            if novo == 'livre':
                # Afasta H da capacidade para o evento de subida não disparar no próprio instante da troca.
                y[3] = ICU - 1e-6
            return novo


        # Método Runge-Kutta (Quarta Ordem) para computar a evolução das EDO´s ao longo do tempo.
        def RK4_lockdown(f, x0, t0, tf, dt, params,N):
            t = np.arange(t0,tf,dt)
            nt = t.size
            nx = x0.size
            x = np.zeros([nx,nt])
            x[:,0] = x0
             
            # Em média, 1 a cada 20 pessoas infectadas com a COVID-19 necessita de uma UTI.
            # O Brasil, durante os piores momentos da pandemia, disponibizou 1 UTI para cada 10000 habitantes do país.
            # Para calcular a capacidade das UTIs ao longo do tempo.
            ICU = (float(values['-uti-'])/10000)*N
            icu = [ICU]*nt

            # Para criar o vetor com dados da variação do número de reprodução básica ao longo do tempo.
            r0 = params["R0"]
            rt = [r0]
            # Valor de tau usado em cada passo (a solução de referência segue os mesmos lockdowns).
            taus = []
            icu_condicao = float(values['-icu-condicao-'])

            # Para calcular a porcentagem de transmisoes que se deve reduzir para controlar uma epidemia.
            tau = params["tau"]
            lockdown_times = 0
            k = 0
            duracao_lockdown = float(values['-duracao-lockdown-'])
            while k < nt-1:
                # Condições para um lockdown de emergência seja acionado.
                if x[3,k] > icu_condicao*ICU and values['-lockdown-'] == 'Yes':
                    lockdown_times += 1
                    count = 1
                    # Um Lockdown de emergência dura 1 mês nessa simulação. 
                    while count < duracao_lockdown/dt:
                        if  k == nt-1:
                            break
                        tau = 0.7
                        taus.append(tau)
                        k1 = dt*f(t[k],x[:,k], tau)
                        k2 = dt*f(t[k] + dt/2, x[:,k] + k1/2, tau)
                        k3 = dt*f(t[k] + dt/2, x[:,k] + k2/2, tau)
                        k4 = dt*f(t[k] + dt, x[:,k] + k3, tau)

                        dx = (k1+2*k2+2*k3+k4)/6
                
                        x[:,k+1] = x[:,k] + dx
                        
                        # Cálculo da variação do número básico de reprodução ao longo do tempo.
                        rt.append(r0*((1-tau)*x[0,k]/N))
                        count += 1
                        k += 1
                else:       
                    tau = params["tau"]
                    taus.append(tau)

                    k1 = dt*f(t[k],x[:,k], tau)
                    k2 = dt*f(t[k] + dt/2, x[:,k] + k1/2, tau)
                    k3 = dt*f(t[k] + dt/2, x[:,k] + k2/2, tau)
                    k4 = dt*f(t[k] + dt, x[:,k] + k3, tau)

                    dx = (k1+2*k2+2*k3+k4)/6

                    x[:,k+1] = x[:,k] + dx

                    # Cálculo da variação do número de reprodução ao longo do tempo.
                    rt.append(r0*((1-tau)*x[0,k]/N))    
                    k += 1   
            return x, t, rt, icu, lockdown_times, taus

        # Método euler para computar a evolução das EDO´s ao longo do tempo.
        def euler_lockdown(f, x0, t0, tf, dt, params, N):
            t = np.arange(t0,tf,dt)
            nt = t.size
            nx = x0.size
            x = np.zeros([nx,nt])
            
            x[:,0] = x0
             
            # Em média, 1 a cada 20 pessoas infectadas com a COVID-19 necessita de uma UTI.
            # O Brasil, durante os piores momentos da pandemia, disponibizou 1 UTI para cada 10000 habitantes do país.
            # Para calcular a capacidade das UTIs ao longo do tempo.
            ICU = (float(values['-uti-'])/10000)*N
            icu = [ICU]*nt

            # Para criar o vetor com dados da variação do número de reprodução básica ao longo do tempo.
            r0 = params["R0"]
            rt = [r0]
            # Valor de tau usado em cada passo (a solução de referência segue os mesmos lockdowns).
            taus = []
            icu_condicao = float(values['-icu-condicao-'])

            # Para calcular a porcentagem de transmisoes que se deve reduzir para controlar uma epidemia.
            tau = params["tau"]

            k = 0
            duracao_lockdown = float(values['-duracao-lockdown-'])
            lockdown_times = 0
            while k < nt-1:
                # Condições para um lockdown de emergência seja acionado.
                if x[3,k] > icu_condicao*ICU and values['-lockdown-'] == 'Yes':
                    lockdown_times += 1
                    count = 1
                    # Um Lockdown de emergência dura 1 mês nessa simulação. 
                    while count < duracao_lockdown/dt:
                        if  k == nt-1:
                            break
                        tau = 0.7
                        taus.append(tau)
                        x[:,k+1] = (x[:,k] + dt*f(t[k],x[:,k],tau))

                        # Cálculo da variação do número básico de reprodução ao longo do tempo.
                        rt.append(r0*((1-tau)*x[0,k]/N))
                        count += 1
                        k += 1
                else:       
                    tau = params["tau"]
                    taus.append(tau)
                    x[:,k+1] = (x[:,k] + dt*f(t[k],x[:,k], tau))
                    # Cálculo da variação do número de reprodução ao longo do tempo.
                    rt.append(r0*((1-tau)*x[0,k]/N))
                    k += 1
            return x, t, rt, icu, lockdown_times, taus

        # Método STIFF: BDF de segunda ordem (VODE/SciPy) para computar a evolução das EDO´s ao longo do tempo.
        def stiff_lockdown(f, x0, t0, tf, dt, params, N):
            t = np.arange(t0,tf,dt)
            nt = t.size
            nx = x0.size
            x = np.zeros([nx,nt])

            x[:,0] = x0

            # Em média, 1 a cada 20 pessoas infectadas com a COVID-19 necessita de uma UTI.
            # O Brasil, durante os piores momentos da pandemia, disponibizou 1 UTI para cada 10000 habitantes do país.
            # Para calcular a capacidade das UTIs ao longo do tempo.
            ICU = (float(values['-uti-'])/10000)*N
            icu = [ICU]*nt

            # Para criar o vetor com dados da variação do número de reprodução básica ao longo do tempo.
            r0 = params["R0"]
            rt = [r0]
            # Valor de tau usado em cada passo (a solução de referência segue os mesmos lockdowns).
            taus = []
            icu_condicao = float(values['-icu-condicao-'])

            # Para calcular a porcentagem de transmisoes que se deve reduzir para controlar uma epidemia.
            tau = params["tau"]

            # Integrador BDF de segunda ordem (VODE/SciPy) - passo interno adaptativo, ordem máxima 2.
            # O lado direito segue o regime da capacidade das UTIs (livre, lotado ou deslizante); as trocas de
            # regime dentro do passo são localizadas por busca de raiz (brentq), como na solução de referência.
            solver = ode(lambda tt, xx, tau, regime: campo_regime(f, tt, xx, tau, regime))
            # atol precisa ser pequeno: no início E, I e H valem frações de pessoa, e um erro relativo nessa fase
            # cresce com a fase exponencial e vira um atraso no pico da epidemia.
            solver.set_integrator('vode', method='bdf', order=2, rtol=1e-8, atol=1e-9, nsteps=5000)
            regime = regime_inicial(f, ICU, t[0], x0, tau)
            config_atual = None

            def passo_bdf2(k, tau):
                nonlocal regime, config_atual
                t_a, y_a = t[k], x[:,k].copy()
                vacina = params["VacinaAtiva"] and t[k] >= params["TempoInicioVacinacao"]
                for _ in range(1000):
                    # Reinicia o integrador só quando tau, o regime ou a fase da vacinação mudam,
                    # para preservar o histórico multipasso do BDF nos demais passos.
                    if (tau, regime, vacina) != config_atual:
                        solver.set_initial_value(y_a, t_a).set_f_params(tau, regime)
                        config_atual = (tau, regime, vacina)
                    y_b = solver.integrate(t[k+1])
                    if not solver.successful():
                        raise RuntimeError(f'BDF2 (VODE) falhou em t = {t_a}')
                    g_b = evento_regime(f, ICU, t[k+1], y_b, tau, regime)
                    if g_b < 0:
                        x[:,k+1] = y_b
                        return

                    # Troca de regime dentro do passo: localiza o instante reintegrando a partir de t_a.
                    g_a = evento_regime(f, ICU, t_a, y_a, tau, regime)
                    def g(s):
                        if s <= t_a:
                            return g_a
                        if s >= t[k+1]:
                            return g_b
                        solver.set_initial_value(y_a, t_a).set_f_params(tau, regime)
                        return evento_regime(f, ICU, s, solver.integrate(s), tau, regime)
                    t_e = t_a if g_a >= 0 else brentq(g, t_a, t[k+1], xtol=1e-6)
                    if t_e > t_a:
                        solver.set_initial_value(y_a, t_a).set_f_params(tau, regime)
                        y_a = solver.integrate(t_e).copy()
                        t_a = t_e
                    regime = trocar_regime(f, ICU, t_a, y_a, tau, regime)
                    config_atual = None
                raise RuntimeError(f'BDF2 (VODE): trocas de regime demais perto de t = {t_a}')

            k = 0
            duracao_lockdown = float(values['-duracao-lockdown-'])
            lockdown_times = 0
            while k < nt-1:
                # Condições para um lockdown de emergência seja acionado.
                if x[3,k] > icu_condicao*ICU and values['-lockdown-'] == 'Yes':
                    lockdown_times += 1
                    count = 1
                    # Um Lockdown de emergência dura 1 mês nessa simulação.
                    while count < duracao_lockdown/dt:
                        if  k == nt-1:
                            break
                        tau = 0.7
                        taus.append(tau)
                        passo_bdf2(k, tau)

                        # Cálculo da variação do número básico de reprodução ao longo do tempo.
                        rt.append(r0*((1-tau)*x[0,k]/N))
                        count += 1
                        k += 1
                else:
                    tau = params["tau"]
                    taus.append(tau)
                    passo_bdf2(k, tau)

                    # Cálculo da variação do número de reprodução ao longo do tempo.
                    rt.append(r0*((1-tau)*x[0,k]/N))
                    k += 1
            return x, t, rt, icu, lockdown_times, taus


        # Solução de referência para medir o erro do método numérico escolhido.
        # Usa LSODA (o mesmo integrador do odeint) com tolerâncias apertadas, a mesma sequência de tau
        # (lockdowns) do método escolhido e os regimes da capacidade das UTIs, com as trocas de regime
        # localizadas por detecção de eventos.
        def solucao_referencia(f, x0, t, taus, ICU, params):
            nt = t.size
            x_ref = np.zeros([x0.size, nt])
            x_ref[:,0] = x0
            regime = regime_inicial(f, ICU, t[0], x0, taus[0])

            # Trechos com tau constante, divididos também no início da vacinação (descontinuidade no tempo).
            cortes = [0] + [k for k in range(1, nt-1) if taus[k] != taus[k-1]]
            if params["VacinaAtiva"]:
                kv = int(np.searchsorted(t, params["TempoInicioVacinacao"]))
                if 0 < kv < nt-1:
                    cortes.append(kv)
            cortes = sorted(set(cortes)) + [nt-1]

            for a, b in zip(cortes[:-1], cortes[1:]):
                tau = taus[a]
                t_atual, y = t[a], x_ref[:,a].copy()
                k_prox = a + 1  # Próximo ponto da malha a ser preenchido.
                if evento_regime(f, ICU, t_atual, y, tau, regime) >= 0:
                    regime = trocar_regime(f, ICU, t_atual, y, tau, regime)
                n_eventos = 0
                while k_prox <= b:
                    campo = lambda tt, xx, tau=tau, regime=regime: campo_regime(f, tt, xx, tau, regime)
                    evento = lambda tt, xx, tau=tau, regime=regime: evento_regime(f, ICU, tt, xx, tau, regime)
                    evento.terminal, evento.direction = True, 1
                    sol = solve_ivp(campo, (t_atual, t[b]), y, method='LSODA', t_eval=t[k_prox:b+1],
                                    events=evento, rtol=1e-10, atol=1e-8)
                    if sol.status == -1:
                        raise RuntimeError(f'Solução de referência falhou em t = {t_atual}: {sol.message}')
                    # sol.t fica vazio (lista) quando o evento ocorre antes do próximo ponto da malha.
                    n = len(sol.t)
                    if n > 0:
                        x_ref[:, k_prox:k_prox + n] = sol.y
                    k_prox += n
                    if sol.status == 0:
                        break

                    # Evento: troca de regime na capacidade das UTIs.
                    t_atual, y = sol.t_events[0][0], sol.y_events[0][0].copy()
                    regime = trocar_regime(f, ICU, t_atual, y, tau, regime)
                    n_eventos += 1
                    if n_eventos > 1000:
                        raise RuntimeError(f'Solução de referência: trocas de regime demais perto de t = {t_atual}')
            return x_ref


        # Helpers de Vacinação
        vacinacao_ativa = values['-vacinacao-'] == 'Yes'
        tempo_inicio_vacinacao_anos = float(values['-tempo-vacinacao-']) if vacinacao_ativa else 0.0
        tempo_inicio_vacinacao_dias = tempo_inicio_vacinacao_anos * 365


        # Parâmetros - Tempo em dias
        t_incubacao = float(values['-incubacao-']) # 5.1
        t_infeccao = float(values['-infeccao-']) # 3.3
        t_imunidade_natural = float(values['-imunidadeNatural-']) # 365
        t_imunidade_vacinados = float(values['-imunidadeVacinados-']) 

        # Parâmetros - Taxas diárias
        tx_internacao = float(values['-internacao-']) # 0.05
        tx_mortalidade_hospitalizados = float(values['-mortalidadeHospitalizados-']); #30% dos hospitalizados falecem
        tx_mortalidade_infectados = float(values['-mortalidadeInfectados-']); 
        tx_mortalidade_natural = float(values['-mortalidadeNatural-']); 
        tx_efetividade = float(values['-taxa-efetividade-']) if vacinacao_ativa else 0.0
        tx_vacinacao = float(values['-taxa-vacinacao-']) if vacinacao_ativa else 0.0 # 0.2% da população é vacinada por dia
        tx_suscetiveis_externos = float(values['-suscetíveisExternos-'])
        tx_expostos_externos = float(values['-expostosExternos-'])
        tx_recuperados_externos = float(values['-recuperadosExternos-'])
                
        # Número de Reprodução Básica.
        R0 = float(values['-repr-']) # 2.5

        # População
        N = int(values['-popsize-']) # 20000000

        # Aumento Populacional - diário
        nascimentos = float(values['-nascimentos-'])
        imigracao = float(values['-imigracao-'])

        # Nivel de distanciamento social.
        # 0.0 - Interação social sem restrições;
        # 0.7 - Lockdown(na prática);
        # 1.0 - Isolamento total(ideal).
        u = float(values['-distance-']) # 0.2

        ICU = (float(values['-uti-'])/10000)*N

        # Parâmetros da modelagem SEIHRVS.
        params = {'R0': R0, 'Sigma': nascimentos , 'Epsillon': imigracao, 'ZetaS': tx_suscetiveis_externos, 'ZetaE': tx_expostos_externos, 'zetaR': tx_recuperados_externos , 'Alpha': 1/t_incubacao, 'Beta': R0/t_infeccao,'GammaI':1/t_infeccao, 'Delta':tx_internacao, 'GammaH':(1-tx_mortalidade_hospitalizados), 'Pi':tx_mortalidade_natural, 'MuI':tx_mortalidade_infectados, 'MuH':tx_mortalidade_hospitalizados, 'OmegaR':1/(t_imunidade_natural*365), 'OmegaV':1/(t_imunidade_vacinados*365),'v': tx_vacinacao, 'e': tx_efetividade,'tau': u, 'VacinaAtiva': vacinacao_ativa, 'TempoInicioVacinacao': tempo_inicio_vacinacao_dias}

        f = lambda t, x, u, sobrecapacidade=None : SEIHRVS_MODEL(x, t, params, N, u, ICU, sobrecapacidade)

        # Condições iniciais do modelo.
        e0 = int(values['-E0-'])
        i0 = int(values['-I0-'])
        h0 = int(values['-H0-'])
        r0 = int(values['-R0-'])
        v0 = int(values['-V0-'])
        s0 = N - e0 -i0 - r0 - h0 - v0
        SEIHRVS_0 = np.array([s0,e0,i0,h0,r0,v0, 0.0, 0.0])

        # Tempo de simulação e passo.
        t0 = 0
        tf = 365*int(values['-time-'])
        dt = 1

        if values['-numerical-methods-'] == "Runge-Kutta":
            # Cálculo de Runge-Kutta.
            x,t,rt,icu,lockdown_times,taus = RK4_lockdown(f, SEIHRVS_0, t0, tf, dt, params, N)
        elif values['-numerical-methods-'] == "STIFF":
            # Cálculo por BDF de segunda ordem (VODE/SciPy).
            x,t,rt,icu,lockdown_times,taus = stiff_lockdown(f, SEIHRVS_0, t0, tf, dt, params, N)
        else:
            # Cálculo de Euler
            x,t,rt,icu,lockdown_times,taus = euler_lockdown(f, SEIHRVS_0, t0, tf, dt, params, N)

        # Solução de referência (LSODA com eventos), com os mesmos lockdowns do método escolhido.
        x_ref = solucao_referencia(f, SEIHRVS_0, t, taus, ICU, params)
                                  
        symbol = ['S','E','I','H','R','V']

        print("RELATÓRIO DE SIMULAÇÃO")
        print("==================================================")
        print("Vacinação: ", vacinacao_ativa)
        if(vacinacao_ativa):
            print("Tempo de início da vacinação: {:.1f} anos".format(tempo_inicio_vacinacao_anos))
        print("LockDown: ", values['-lockdown-'])
        if(values['-lockdown-'] == 'Yes'):
            print("Número de vezes que o lockdown foi acionado: ", lockdown_times)
        print('Quantidade de leitos de UTI disponíveis: {}'.format(int(ICU)))
        print("Passo utilizado(h): ", dt)
        print("Método numérico utilizado: ", values['-numerical-methods-'])
        if values['-numerical-methods-'] == "STIFF":
            print("Integrador: BDF ordem 2 (VODE/SciPy), passo interno adaptativo, trocas de regime das UTIs por eventos")
        print("==================================================")
        print('Número total de óbitos ao final da simulação: ', x[6,-1] + x[7,-1]) 
        print('Número total de óbitos por causas naturais ao final da simulação: ', x[6,-1])
        print('Número total de óbitos pela doença ao final da simulação: ', x[7,-1])
        print('Número total de pessoas recuperadas ao final da simulação: ', x[4,-1])
        print('Número total de pessoas vacinadas ao final da simulação: ', x[5,-1])
        print('Número total de pessoas suscetíveis ao final da simulação: ', x[0,-1])
        print('Número total de pessoas expostas ao final da simulação: ', x[1,-1])
        print('Número total de pessoas infectadas ao final da simulação: ', x[2,-1])
        print('Número de pessoas hospitalizadas ao final da simulação: ', x[3,-1])
        print('Pico de pessoas hospitalizadas: ', max(x[3,:]))
        print("==================================================")
            
        # Calculando o erro quadrático médio do método numérico selecionado em relação à solução de referência.
        msr = []
        for i in range(0, 6):
            msr.append(((x[i,:] - x_ref[i,:])**2).mean(axis=None))
            txt = f"Erro quadrático médio {symbol[i]} - {values['-numerical-methods-']} e referência (LSODA): "
            print(txt, msr[i])
        print()
        
        # Plotando dos gráficos.
        fig, ax = plt.subplots(2, 1)
        
        # Gráfico da simulação epidemiológica SEIHRVS.
        model_name = 'SEIHRVS' if vacinacao_ativa else 'SEIHRS'
        ax[0].set_title(f'Simulação epidemiológica {model_name} - {values['-numerical-methods-']}')
        ax[0].plot(t/365, x[0,:], 'r', label = 'S')
        ax[0].plot(t/365, x[1,:], 'g', label = 'E')
        ax[0].plot(t/365, x[2,:], 'b', label = 'I')
        ax[0].plot(t/365, x[3,:], 'm', label = 'H')
        ax[0].plot(t/365, x[4,:], 'y', label = 'R')
        if vacinacao_ativa:
            ax[0].plot(t/365, x[5,:], 'c', label = 'V')
            ax[0].axvline(tempo_inicio_vacinacao_anos, color='k', linestyle='-', linewidth=1.5, label='Início da vacinação')
        ax[0].plot(t/365, icu, linestyle = '--', color = 'k', label = 'Capacidade das UTIs')
        ax[0].set_xlabel('tempo(anos)')
        ax[0].set_ylabel('População')
        ax[0].grid()
        ax[0].legend()
        
        # Gráfico da variação do número de reprodução básica ao longo do tempo.
        ax[1].set_title('Número de Reprodução')
        ax[1].plot(t/365, rt, label = 'RT')
        ax[1].set_xlabel('tempo(anos)')
        ax[1].grid()
        ax[1].legend()
        

        plt.subplots_adjust(hspace=0.8)
        plt.show()

window.close()