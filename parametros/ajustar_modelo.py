"""
Ajuste do modelo SEIHRVS por mínimos quadrados com limites nas variáveis.

    Fixos, da tabela do texto: p_H, p_C, d, demais atrasos, e_1, e_2 e ICU
    Etapa A (lineares, equações normais x = (A^T A)^-1 A^T b, com limites):
        tau_IH                <- dias entre início dos sintomas e internação, no SINAN
        c_1, v_1,g, c_2,g     <- doses mensais por grupo, nascimentos e população por faixa
    Etapa B (não linear, least_squares com limites):
        R0, s0                <- casos diários do SINAN, simulando o sistema inteiro
    p_H, p_C e d também são estimados no SINAN e no SIH, só para comparar com a tabela.

Hipóteses declaradas (os dados não permitem medi-las):
    - rho = 1: todo caso é notificado, nos casos, nas internações e nos importados.
    - Importação: cada mês com casos é um evento no dia 15, com n_k infecciosos.
    - Doses não seletivas: a taxa por pessoa não depende de a pessoa estar imune.
      Logo v = doses / população do grupo, e o fluxo do modelo continua v * S.
    - Coorte do grupo 1 só com mortalidade natural: lambda desprezível na sobrevivência aos 12 meses.
    - Tetraviral na D2 do grupo 1 só de 2023 em diante; antes disso não há dado.
    - s0: mesma fração suscetível em 0-59 anos em T0; o restante começa em V.
    - Todos os maiores de 60 anos começam imunes, em R.
    - DT_SIN_PRI é tratado como a entrada em I; tau_IH só é medido nos internados.
    - Sem doses diárias: as taxas são por dia, mas ajustadas às doses mensais.
    - Janela até março de 2020: depois dela a pandemia altera o contato, e o modelo não tem sinal de controle.

    python parametros/ajustar_modelo.py
"""

import io
import os
import re
import sys

import numpy as np
import pandas as pd
from itertools import product

from scipy.optimize import least_squares
from scipy.stats import qmc

RAIZ = os.path.dirname(os.path.abspath(__file__))
PROJETO = os.path.dirname(RAIZ)
sys.path[:0] = [PROJETO, RAIZ]

import calcular_parametros_hospitalares as hosp
from SEIHRVS import IDX_C, IDX_D_DIS, IDX_S, IDX_V, condicao_inicial, simular

# =============================================================================
# CONFIGURACOES
# =============================================================================

DADOS = os.path.join(PROJETO, "dados")
ARQUIVO_EVENTOS = os.path.join(DADOS, "resultados", "importacao", "importacao_eventos.csv")
SAIDA_PARAMETROS = os.path.join(DADOS, "resultados", "ajustes", "ajuste_nacional_parametros.csv")
SAIDA_CENARIOS = os.path.join(DADOS, "resultados", "ajustes", "ajuste_nacional_cenarios.csv")
SAIDA_CURVAS = os.path.join(DADOS, "resultados", "ajustes", "ajuste_nacional_curvas.csv")
SAIDA_FIGURA = os.path.join(RAIZ, "ajuste_mq.png")

T0 = pd.Timestamp("2016-01-01")       # primeiro mês com doses
FIM = pd.Timestamp("2020-03-31")      # último mês antes da queda abrupta de abril de 2020
FIM_TREINO = pd.Timestamp("2019-06-30")
ANOS_HOSP = list(range(2007, 2027))
DOSES_FIM = "2026-08"                  # sensibilidade: até o último mês com nascimentos 12 meses antes

# Valores da tabela de parâmetros do texto
FIXOS = {
    "tau_E": 12.5, "tau_IR": 8.0, "tau_HC": 6.7, "tau_HR": 4.47,
    "tau_HD": 4.8, "tau_CR": 14.0, "tau_CD": 7.9, "tau_D1": 365.0, "tau_v2_1": 90.0,
    "tau_v2_2": 30.0, "e1": 0.93, "e2": 0.571, "ICU": 2.43, "pH": 0.2, "pC": 0.032, "d": 0.0804,
}

# Limites: R0 do texto (12-18); s0 é fração
LIMITES = {"R0": (12.0, 18.0), "s0": (0.0, 1.0)}
USA_LOG = {"R0": False, "s0": False}

N_PARTIDAS = 12
PASSO_DIF = 0.01
SEMENTE = 2026
NOMES_VACINA = ("c1", "v1_2", "v1_3", "c2_1", "c2_2")
LIMITE_VACINA = {"c1": 1.0, "v1_2": np.inf, "v1_3": np.inf, "c2_1": 1.0, "c2_2": 1.0}
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


# =============================================================================
# LEITURA
# =============================================================================

def ler_serie_mensal(caminho):
    """Série 'Mês/AAAA' do TabNet; o mês vem da ordem das linhas dentro de cada ano."""
    bruto = open(caminho, "rb").read()
    valores, posicao = {}, {}
    for linha in io.StringIO(bruto.decode("latin-1")):
        campos = [c.strip().strip('"') for c in linha.split(";")]
        achou = re.match(r"^.+/(\d{4})$", campos[0])
        if achou and campos[1].lstrip("-").isdigit():
            ano = int(achou.group(1))
            posicao[ano] = posicao.get(ano, 0) + 1
            valores[pd.Period(f"{ano}-{posicao[ano]:02d}", "M")] = float(campos[1])
    return pd.Series(valores).sort_index()


def populacao_por_ano():
    pop = {}
    for ano in range(2007, 2026):
        tabela = pd.read_csv(os.path.join(DADOS, "populacao", f"populacao_por_uf_{ano}.csv"), sep=None, engine="python")
        pop[ano] = float(tabela["POP"].sum())
    pop[2026] = pop[2025]  # sem população de 2026: repete 2025
    return pop


def ler_projecao(faixa):
    """Linha Total da projeção do IBGE (edição 2024), por ano."""
    tabela = pd.read_csv(os.path.join(DADOS, "populacao", f"projecao_ibge_idade_{faixa}.csv"), sep=";",
                         encoding="latin-1", skiprows=4, nrows=6)
    total = tabela[tabela.iloc[:, 0] == "Total"].iloc[0, 1:]
    return {int(ano): float(valor) for ano, valor in total.items()}


def populacao_por_faixa():
    """Grupos do modelo: 1 = 0-4, 2 = 5-29, 3 = 30-59 e 60 = 60 anos ou mais."""
    zero, um_a_quatro = ler_projecao("0_anos"), ler_projecao("1_a_4_anos")
    return {1: {a: zero[a] + um_a_quatro[a] for a in zero}, 2: ler_projecao("5_a_29_anos"),
            3: ler_projecao("30_a_59_anos"), 60: ler_projecao("60_anos_ou_mais")}


def carregar_dados():
    series = {
        "nascimentos": ler_serie_mensal(os.path.join(DADOS, "series_mensais", "nascimentos_mensais_brasil.csv")),
        "mortes": ler_serie_mensal(os.path.join(DADOS, "series_mensais", "obitos_mensais_brasil.csv")),
    }
    sinan = hosp.carregar_sinan(ANOS_HOSP)
    sih = hosp.carregar_sih(ANOS_HOSP, hosp.TODAS_UFS)
    eventos = pd.read_csv(ARQUIVO_EVENTOS, sep=";")
    return series, sinan, sih, populacao_por_ano(), eventos, ler_doses_grupos(), populacao_por_faixa()


# =============================================================================
# ETAPA A: PROBLEMAS LINEARES
# =============================================================================

def ajuste_linear(A, y, inf, sup):
    """x = (A^T A)^-1 A^T y nas variáveis livres, as demais no limite, mais os diagnósticos."""
    A, y = np.asarray(A, float), np.asarray(y, float)
    if A.ndim == 1:
        A = A[:, None]
    escala = np.linalg.norm(A, axis=0)
    B = A / escala
    inf_e, sup_e = np.array(inf, float) * escala, np.array(sup, float) * escala

    # Cada variável livre, no limite inferior ou no superior: fica a combinação viável de menor erro
    x, menor = None, np.inf
    for estados in product((None, inf_e, sup_e), repeat=A.shape[1]):
        tentativa = np.zeros(A.shape[1])
        livres = np.array([e is None for e in estados])
        for j, e in enumerate(estados):
            if e is not None:
                tentativa[j] = e[j]
        if not np.isfinite(tentativa).all():
            continue
        if livres.any():
            b = y - B[:, ~livres] @ tentativa[~livres]
            tentativa[livres] = np.linalg.solve(B[:, livres].T @ B[:, livres], B[:, livres].T @ b)
        if (tentativa < inf_e - 1e-9 * np.abs(inf_e)).any() or (tentativa > sup_e + 1e-9 * np.abs(sup_e)).any():
            continue
        erro = float(((y - B @ tentativa) ** 2).sum())
        if erro < menor:
            x, menor = tentativa, erro
    x = x / escala
    res = y - A @ x
    rss = float(res @ res)
    rmse = float(np.sqrt(rss / len(y)))
    gl = max(len(y) - A.shape[1], 1)
    cov = rss / gl * np.linalg.pinv(A.T @ A)
    return {"x": x, "ep": np.sqrt(np.diag(cov)), "rmse": rmse,
            "r2": 1 - rss / float(((y - y.mean()) ** 2).sum()),
            "cond": float(np.linalg.cond(B.T @ B)), "n": len(y),
            "limite": [bool(np.isclose(v, a) or np.isclose(v, b)) for v, a, b in zip(x, inf, sup)]}


def sistemas_clinicos(sinan, sih):
    """(entradas, eventos) mensais de p_H, p_C e d: o mês é o da entrada, então não entram atrasos."""
    inf = sinan[sinan["ID_HOSPIT"].isin([hosp.HOSPIT_SIM, hosp.HOSPIT_NAO])]
    mes_s = inf["DT_SIN_PRI"].dt.to_period("M")
    tab = pd.DataFrame({"n": inf.groupby(mes_s).size(),
                        "h": (inf["ID_HOSPIT"] == hosp.HOSPIT_SIM).groupby(mes_s).sum()}).fillna(0)

    com_uti = sih["UTI_DIAS"] > 0
    mes_h = sih["DT_INTER"].dt.to_period("M")
    tab_h = pd.DataFrame({"h": sih.groupby(mes_h).size(), "u": com_uti.groupby(mes_h).sum(),
                          "o": (com_uti & (sih["MORTE"] == 1)).groupby(mes_h).sum()}).fillna(0)
    tab_u = tab_h[tab_h["u"] > 0]

    return {"pH": (tab["n"], tab["h"]), "pC": (tab_h["h"], tab_h["u"]), "d": (tab_u["u"], tab_u["o"])}


def etapa_clinica(sinan, sih):
    return {nome: ajuste_linear(n, y, [0.0], [1.0]) for nome, (n, y) in sistemas_clinicos(sinan, sih).items()}


def etapa_atraso_internacao(sinan):
    """tau_IH com A = coluna de uns: a solução é a média dos dias, como dias_I1_ate_H do notebook."""
    internados = sinan[sinan["ID_HOSPIT"] == hosp.HOSPIT_SIM]
    dias = (internados["DT_INTERNA"] - internados["DT_SIN_PRI"]).dt.days
    dias = dias[dias.between(*hosp.DIAS_ATE_INTERNACAO)].values.astype(float)
    return ajuste_linear(np.ones(dias.size), dias, [hosp.DIAS_ATE_INTERNACAO[0]], [hosp.DIAS_ATE_INTERNACAO[1]])


def sem_campanha(y, limiar=2.5, janela=13):
    # Mesma regra de vacinacao_sarampo_colab.ipynb: mês / mediana móvel acima do limiar
    return (y / y.rolling(janela, center=True, min_periods=1).median() <= limiar).values


def ler_doses_csv(caminho):
    """2008-2022: UF nas linhas e colunas AAAA/Mes, fora de ordem em alguns anos."""
    df = pd.read_csv(caminho, sep=";", encoding="latin-1")
    df = df[df.iloc[:, 0].notna() & (df.iloc[:, 0].astype(str).str.strip().str.lower() != "total")]
    colunas = [c for c in df.columns[1:] if "/" in c]
    valores = df[colunas].replace("-", 0).apply(pd.to_numeric).sum()
    valores.index = [pd.Period(f"{c.split('/')[0].strip()}-{MESES.index(c.split('/')[1].strip()) + 1:02d}", "M") for c in colunas]
    return valores.groupby(level=0).sum()


def ler_doses_xlsx(caminho):
    """2023-2026: um bloco por ano, com '-' para zero doses."""
    df = pd.read_excel(caminho, header=None)
    valores = df.iloc[3:, 2:].replace("-", 0).apply(pd.to_numeric).sum()
    valores.index = pd.period_range(f"{int(df.iloc[1, 2])}-01", periods=valores.size, freq="M")
    return valores


def ler_doses_grupos():
    """Doses mensais do Brasil por (dose, grupo). D2 do grupo 1 inclui a tetraviral (só há dado de 2023 em diante)."""
    pasta = os.path.join(DADOS, "vacinacao")
    doses = {}
    for dose, grupo in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2)):
        doses[dose, grupo] = pd.concat([
            ler_doses_csv(os.path.join(pasta, f"doses_d{dose}_grupo{grupo}_mensais_2008_2022_sem_campanhas.csv")),
            ler_doses_xlsx(os.path.join(pasta, f"doses_d{dose}_grupo{grupo}_mensais_2023_2026.xlsx"))]).sort_index()
    tetraviral = ler_doses_xlsx(os.path.join(pasta, "doses_tetraviral_d1_grupo1_mensais_2023_2026.xlsx"))
    doses[2, 1] = doses[2, 1].add(tetraviral, fill_value=0.0)
    return doses


def sistemas_vacinais(series, doses, pop_faixas, pi_taxa, ini=T0, fim=FIM):
    """Um sistema por grupo: (coluna do modelo, doses observadas, meses), sem os meses de campanha."""
    meses = pd.period_range(ini, fim, freq="M")
    dias = meses.days_in_month.values.astype(float)
    nasc = series["nascimentos"]
    coorte = np.array([nasc[m - 12] for m in meses]) * np.exp(-pi_taxa * FIXOS["tau_D1"])

    def sistema(coluna, y):
        ok = sem_campanha(pd.Series(y))
        return coluna[ok], y[ok], meses[ok]

    def observado(dose, grupo, atraso=0):
        return doses[dose, grupo].reindex(meses - atraso).values

    pessoas_dia = {g: np.array([pop_faixas[g][m.year] for m in meses]) * dias for g in (2, 3)}
    return {"c1": sistema(coorte, observado(1, 1)),
            "v1_2": sistema(pessoas_dia[2], observado(1, 2)),
            "v1_3": sistema(pessoas_dia[3], observado(1, 3)),
            # Intervalos D1 -> D2: 90 dias no grupo 1 e 30 dias no grupo 2
            "c2_1": sistema(observado(1, 1, 3), observado(2, 1)),
            "c2_2": sistema(observado(1, 2, 1), observado(2, 2))}


def etapa_vacinal(series, doses, pop_faixas, pi_taxa, ini=T0, fim=FIM):
    return {nome: ajuste_linear(x, y, [0.0], [LIMITE_VACINA[nome]])
            for nome, (x, y, _) in sistemas_vacinais(series, doses, pop_faixas, pi_taxa, ini, fim).items()}


# =============================================================================
# ETAPA B: SIMULACAO COMPLETA
# =============================================================================

class Problema:
    """Dados da janela e simulação do modelo para uma escolha de parâmetros."""

    def __init__(self, series, sinan, pop, eventos, vacina, pop_faixas, tau_IH, fim=FIM):
        self.dias = pd.date_range(T0, FIM, freq="D")
        # O último passo da simulação não grava entradas: esse dia fica fora dos resíduos
        self.n_dias = min((pd.Timestamp(fim) - T0).days + 1, self.dias.size - 1)

        janela = pd.period_range(T0, FIM, freq="M")
        self.faixas = tuple(pop_faixas[g][T0.year] for g in (1, 2, 3, 60))
        self.N = sum(self.faixas)
        pop_media = np.mean([pop[m.year] for m in janela])
        dias_total = self.dias.size
        self.base = dict(FIXOS, tau_IH=tau_IH)
        self.base["Lambda"] = float(series["nascimentos"].reindex(janela).sum() / dias_total)
        self.base["Pi"] = float(series["mortes"].reindex(janela).sum() / dias_total / pop_media)
        self.base.update(c1=float(vacina["c1"]["x"][0]),
                         v1_2=float(vacina["v1_2"]["x"][0]), v1_3=float(vacina["v1_3"]["x"][0]),
                         c2_1=float(vacina["c2_1"]["x"][0]), c2_2=float(vacina["c2_2"]["x"][0]))

        self.y_c = self._diario(sinan["DT_SIN_PRI"])
        # Internação: DT_INTERNA se plausível; senão, início dos sintomas + tau_IH
        internados = sinan[sinan["ID_HOSPIT"] == hosp.HOSPIT_SIM]
        atraso = (internados["DT_INTERNA"] - internados["DT_SIN_PRI"]).dt.days
        valida = atraso.between(*hosp.DIAS_ATE_INTERNACAO)
        data = internados["DT_INTERNA"].where(
            valida, internados["DT_SIN_PRI"] + pd.Timedelta(days=round(tau_IH)))
        self.y_h = self._diario(data)

        self.importados = np.zeros(self.dias.size)
        for _, ev in eventos.iterrows():
            dia = pd.Timestamp(int(ev["ano"]), MESES.index(ev["mes"]) + 1, 15)
            if T0 <= dia <= FIM:
                self.importados[(dia - T0).days] += float(ev["n_k"])

    def _diario(self, datas):
        dias = (datas.dropna() - T0).dt.days.values
        dias = dias[(dias >= 0) & (dias < self.dias.size)]
        return np.bincount(dias, minlength=self.dias.size).astype(float)

    def condicao_inicial(self):
        return condicao_inicial(self.faixas, 0.0, 0.0, 0.0, 0.0)

    def simular(self, teta):
        params = dict(self.base)
        params["R0"] = float(teta["R0"])
        s0 = float(teta["s0"])

        x0 = self.condicao_inicial()
        for s, v in zip(IDX_S, IDX_V):
            x0[v] = (1.0 - s0) * x0[s]
            x0[s] = s0 * x0[s]

        fluxos = {}
        x, _, _, icu = simular(x0, 0.0, float(self.dias.size), 1.0, params, self.importados, fluxos)
        return {"c": fluxos["ent_I"], "h": fluxos["ent_H"], "u": fluxos["ent_C"], "x": x, "icu": icu}


class Ajuste:
    """Mínimos quadrados não lineares com limites, em variáveis normalizadas em [0, 1]."""

    def __init__(self, problema, nomes, fixos=None, y_sinteticos=None):
        self.p, self.nomes, self.fixos = problema, nomes, dict(fixos or {})
        self.y_c = problema.y_c if y_sinteticos is None else y_sinteticos
        self.limites = [self._transforma(n, np.array(LIMITES[n])) for n in nomes]

    @staticmethod
    def _transforma(nome, valor):
        return np.log10(valor) if USA_LOG[nome] else valor

    def teta(self, z):
        teta = dict(self.fixos)
        for nome, zi, (a, b) in zip(self.nomes, z, self.limites):
            valor = a + zi * (b - a)
            teta[nome] = 10.0 ** valor if USA_LOG[nome] else valor
        return teta

    def z_de(self, teta):
        return np.array([(self._transforma(n, teta[n]) - a) / (b - a) for n, (a, b) in zip(self.nomes, self.limites)])

    def residuos(self, z):
        sim = self.p.simular(self.teta(z))
        n = self.p.n_dias
        return np.log1p(self.y_c[:n]) - np.log1p(sim["c"][:n])

    def otimizar(self, partidas):
        solucoes = []
        for z0 in partidas:
            sol = least_squares(self.residuos, np.clip(z0, 1e-3, 1 - 1e-3), bounds=(0.0, 1.0), method="trf",
                                diff_step=PASSO_DIF, x_scale=1.0, max_nfev=40)
            solucoes.append(sol)
        solucoes.sort(key=lambda s: s.cost)
        return solucoes

    def partidas(self, n, semente=SEMENTE):
        return qmc.LatinHypercube(len(self.nomes), seed=semente).random(n)


def diagnosticos(ajuste, z):
    """SVD da Jacobiana em log dos parâmetros e covariância aproximada."""
    teta0 = ajuste.teta(z)
    nomes = ajuste.nomes
    log0 = np.array([np.log(max(teta0[n], 1e-12)) for n in nomes])
    r0 = ajuste.residuos(z)

    def r_log(log_t):
        teta = dict(teta0)
        teta.update({n: float(np.exp(v)) for n, v in zip(nomes, log_t)})
        return ajuste.residuos(ajuste.z_de(teta))

    J = np.zeros((r0.size, len(nomes)))
    for j in range(len(nomes)):
        h = np.zeros(len(nomes))
        h[j] = 0.02
        J[:, j] = (r_log(log0 + h) - r_log(log0 - h)) / 0.04
    _, sv, Vt = np.linalg.svd(J, full_matrices=False)
    sigma2 = float(r0 @ r0) / max(r0.size - len(nomes), 1)
    ep_log = np.sqrt(np.diag(sigma2 * np.linalg.pinv(J.T @ J)))
    r1 = r0[:ajuste.p.n_dias]
    return {"sv": sv, "Vt": Vt, "ep_log": ep_log, "sigma2": sigma2,
            "autocorr_1": float(np.corrcoef(r1[:-1], r1[1:])[0, 1])}


def r2_rmse(y, previsto):
    erro = float(((y - previsto) ** 2).sum())
    return 1 - erro / float(((y - y.mean()) ** 2).sum()), float(np.sqrt(erro / y.size))


def resumo(problema, teta, y=None):
    sim = problema.simular(teta)
    y = problema.y_c if y is None else y
    n = problema.n_dias
    r2, rmse = r2_rmse(y[:n], sim["c"][:n])
    r2_log, rmse_log = r2_rmse(np.log1p(y[:n]), np.log1p(sim["c"][:n]))
    r2_h, rmse_h = r2_rmse(problema.y_h[:n], sim["h"][:n])
    return {"r2": r2, "rmse": rmse, "r2_log": r2_log, "rmse_log": rmse_log, "r2_intern": r2_h, "rmse_intern": rmse_h,
            "sse_log": float(((np.log1p(y[:n]) - np.log1p(sim["c"][:n])) ** 2).sum()),
            "casos_modelo": float(sim["c"][:n].sum()), "casos_obs": float(y[:n].sum()),
            "dia_pico_modelo": int(sim["c"][:n].argmax()), "dia_pico_obs": int(y[:n].argmax()),
            "pico_modelo": float(sim["c"][:n].max()), "pico_obs": float(y[:n].max()),
            "R0_s0": float(teta["R0"] * teta["s0"])}


# =============================================================================
# EXECUCAO
# =============================================================================

def imprimir_linear(nome, a, rotulos):
    valores = ", ".join(f"{r}={v:.5g} (EP {e:.2g}{', NO LIMITE' if l else ''})"
                        for r, v, e, l in zip(rotulos, a["x"], a["ep"], a["limite"]))
    print(f"  {nome}: {valores} | R2={a['r2']:.3f} RMSE={a['rmse']:.4g} n={a['n']} cond={a['cond']:.3g}")


def main():
    print("Lendo dados")
    series, sinan, sih, pop, eventos, doses, pop_faixas = carregar_dados()
    pi_ini = float(series["mortes"].reindex(pd.period_range(T0, FIM, freq="M")).sum()
                   / ((FIM - T0).days + 1) / pop[2018])

    print("\nComparação com a tabela: p_H, p_C e d estimados, não usados no ajuste")
    clinica = etapa_clinica(sinan, sih)
    for nome in ("pH", "pC", "d"):
        imprimir_linear(f"{nome} (tabela {FIXOS[nome]})", clinica[nome], [nome])

    print("\nEtapa A: mínimos quadrados lineares com limites")
    atraso = etapa_atraso_internacao(sinan)
    tau_IH = float(atraso["x"][0])
    imprimir_linear("tau_IH (dias)", atraso, ["tau_IH"])
    vacina = etapa_vacinal(series, doses, pop_faixas, pi_ini)
    for nome in NOMES_VACINA:
        imprimir_linear(f"{nome} ({T0.year}-{FIM.year})", vacina[nome], [nome])
    longo = etapa_vacinal(series, doses, pop_faixas, pi_ini, fim=pd.Period(DOSES_FIM, "M").end_time)
    for nome in NOMES_VACINA:
        imprimir_linear(f"{nome} (até {DOSES_FIM})", longo[nome], [nome])

    problema = Problema(series, sinan, pop, eventos, vacina, pop_faixas, tau_IH)
    print(f"\nLambda={problema.base['Lambda']:.1f}/dia Pi={problema.base['Pi']:.3e}/dia N={problema.N}")
    print(f"Dias={problema.n_dias}, casos={problema.y_c[:problema.n_dias].sum():.0f}, "
          f"internações={problema.y_h[:problema.n_dias].sum():.0f}, importados={problema.importados.sum():.0f}")

    linhas_cen = []

    def registrar(cenario, teta, extra=None):
        linha = {"cenario": cenario, **{k: teta.get(k) for k in ("R0", "s0")}}
        linha.update(resumo(problema, teta))
        linha.update(extra or {})
        linhas_cen.append(linha)
        print(f"  {cenario}: R0={teta['R0']:.2f} s0={teta['s0']:.4f} "
              f"R0*s0={teta['R0'] * teta['s0']:.3f} R2={linha['r2']:.3f} RMSE={linha['rmse']:.1f} SSE={linha['sse_log']:.1f} "
              f"casos={linha['casos_modelo']:.0f}/{linha['casos_obs']:.0f} "
              f"pico dia={linha['dia_pico_modelo']}/{linha['dia_pico_obs']}")

    nomes = ["R0", "s0"]

    print("\nTeste sintético (dados Poisson gerados pelo modelo, parâmetros conhecidos)")
    verdade = {"R0": 15.0, "s0": 0.08}
    rng = np.random.default_rng(SEMENTE)
    y_sint = rng.poisson(problema.simular(verdade)["c"]).astype(float)
    aj_sint = Ajuste(problema, nomes, y_sinteticos=y_sint)
    sint = aj_sint.otimizar(aj_sint.partidas(N_PARTIDAS))[0]
    t_sint = aj_sint.teta(sint.x)
    print(f"  verdade: {verdade}, R0*s0={verdade['R0'] * verdade['s0']:.3f}")
    print(f"  obtido: R0={t_sint['R0']:.3f} s0={t_sint['s0']:.4f} "
          f"R0*s0={t_sint['R0'] * t_sint['s0']:.3f}")
    dg_sint = diagnosticos(aj_sint, sint.x)
    print(f"  valores singulares: {np.round(dg_sint['sv'], 3)}")

    print("\nEtapa B: casos diários")
    aj_b = Ajuste(problema, nomes)
    sols = aj_b.otimizar(aj_b.partidas(N_PARTIDAS))
    print("  custos das partidas:", np.round([2 * s.cost for s in sols], 1))
    print("  soluções distintas:")
    for s in sols[:5]:
        t = aj_b.teta(s.x)
        print(f"    SSE={2 * s.cost:.1f} R0={t['R0']:.2f} s0={t['s0']:.4f} status={s.status}")
    teta_b = aj_b.teta(sols[0].x)
    registrar("B: casos", teta_b)
    dg_b = diagnosticos(aj_b, sols[0].x)
    print(f"  valores singulares={np.round(dg_b['sv'], 3)}")
    print(f"  direções (ln R0, ln s0):\n{np.round(dg_b['Vt'], 3)}")
    print(f"  autocorr. lag 1 dos resíduos={dg_b['autocorr_1']:.3f} sigma2={dg_b['sigma2']:.3f}")

    print("\nPerfil de R0")
    for r0 in (12.0, 15.0, 18.0):
        aj = Ajuste(problema, ["s0"], fixos={"R0": r0})
        registrar(f"perfil R0={r0:.0f}", aj.teta(aj.otimizar(aj.partidas(8))[0].x))

    sim_b = problema.simular(teta_b)

    print("\nSensibilidade: p_H, p_C e d estimados na Etapa A, no lugar dos valores da tabela")
    prob_est = Problema(series, sinan, pop, eventos, vacina, pop_faixas, tau_IH)
    prob_est.base.update({nome: float(clinica[nome]["x"][0]) for nome in ("pH", "pC", "d")})
    aj_est = Ajuste(prob_est, nomes)
    teta_est = aj_est.teta(aj_est.otimizar(aj_est.partidas(N_PARTIDAS))[0].x)
    linha_est = {"cenario": "frações estimadas", **teta_est, **resumo(prob_est, teta_est)}
    linhas_cen.append(linha_est)
    print(f"  R0={teta_est['R0']:.2f} s0={teta_est['s0']:.4f} R2={linha_est['r2']:.3f} RMSE={linha_est['rmse']:.1f}")

    print(f"\nValidação: ajuste até {FIM_TREINO.date()}, previsão até {FIM.date()}")
    prob_t = Problema(series, sinan, pop, eventos, vacina, pop_faixas, tau_IH, fim=FIM_TREINO)
    aj_t = Ajuste(prob_t, nomes)
    teta_t = aj_t.teta(aj_t.otimizar(aj_t.partidas(N_PARTIDAS))[0].x)
    sim_t = prob_t.simular(teta_t)
    n_t = prob_t.n_dias
    sse_fora = float(((np.log1p(problema.y_c[n_t:problema.n_dias]) - np.log1p(sim_t["c"][n_t:problema.n_dias])) ** 2).sum())
    sse_fora_b = float(((np.log1p(problema.y_c[n_t:problema.n_dias]) - np.log1p(sim_b["c"][n_t:problema.n_dias])) ** 2).sum())
    print(f"  R0={teta_t['R0']:.2f} s0={teta_t['s0']:.4f}")
    print(f"  casos fora do treino: modelo={sim_t['c'][n_t:problema.n_dias].sum():.0f} "
          f"obs={problema.y_c[n_t:problema.n_dias].sum():.0f}")
    print(f"  SSE fora do treino={sse_fora:.1f} (ajuste completo no mesmo trecho={sse_fora_b:.1f})")

    # Conferências no ajuste final
    obitos_modelo = sim_b["x"][IDX_D_DIS, -1]
    na_janela = sinan["DT_SIN_PRI"].between(T0, FIM)
    print("\nConferências (ajuste B)")
    print(f"  Óbitos: modelo={obitos_modelo:.1f}; SINAN={(sinan.loc[na_janela, 'EVOLUCAO'] == hosp.EVOLUCAO_OBITO_AGRAVO).sum()}; "
          f"SIH={(sih.loc[sih['DT_INTER'].between(T0, FIM), 'MORTE'] == 1).sum()}")
    print(f"  Admissões UTI: modelo={sim_b['u'].sum():.1f}; SIH="
          f"{(sih.loc[sih['DT_INTER'].between(T0, FIM), 'UTI_DIAS'] > 0).sum()}")
    print(f"  Pico em C={sim_b['x'][IDX_C].max():.1f}; leitos={sim_b['icu'].min():.0f}")
    print(f"  Internações: modelo={sim_b['h'][:problema.n_dias].sum():.0f}; SINAN={problema.y_h[:problema.n_dias].sum():.0f}")

    # Saídas
    linhas = [{"parametro": "tau_IH", "valor": tau_IH, "erro_padrao": atraso["ep"][0], "etapa": "A",
               "n": atraso["n"], "no_limite": atraso["limite"][0], "rmse": atraso["rmse"]}]
    for nome in ("pH", "pC", "d"):
        a = clinica[nome]
        linhas.append({"parametro": nome, "valor": FIXOS[nome], "etapa": "tabela"})
        linhas.append({"parametro": nome, "valor": a["x"][0], "erro_padrao": a["ep"][0], "etapa": "estimado, não usado",
                       "n": a["n"], "no_limite": a["limite"][0], "r2": a["r2"], "rmse": a["rmse"]})
    for nome in NOMES_VACINA:
        a = vacina[nome]
        linhas.append({"parametro": nome, "valor": a["x"][0], "erro_padrao": a["ep"][0], "etapa": "A",
                       "n": a["n"], "no_limite": a["limite"][0], "r2": a["r2"], "rmse": a["rmse"]})
    linhas += [{"parametro": "Lambda", "valor": problema.base["Lambda"], "etapa": "dados"},
               {"parametro": "Pi", "valor": problema.base["Pi"], "etapa": "dados"}]
    for j, n in enumerate(nomes):
        a, b = LIMITES[n]
        linhas.append({"parametro": n, "valor": teta_b[n], "erro_padrao": teta_b[n] * dg_b["ep_log"][j], "etapa": "B",
                       "n": problema.n_dias, "no_limite": bool(np.isclose(teta_b[n], a, rtol=1e-3) or np.isclose(teta_b[n], b, rtol=1e-3))})
    pd.DataFrame(linhas).to_csv(SAIDA_PARAMETROS, sep=";", index=False)
    pd.DataFrame(linhas_cen).to_csv(SAIDA_CENARIOS, sep=";", index=False)

    n = problema.n_dias
    datas = T0 + pd.to_timedelta(np.arange(n), unit="D")
    pd.DataFrame({"dia": datas, "casos_obs": problema.y_c[:n], "casos_B": sim_b["c"][:n],
                  "casos_treino": sim_t["c"][:n], "intern_obs": problema.y_h[:n],
                  "intern_B": sim_b["h"][:n]}).to_csv(SAIDA_CURVAS, sep=";", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(datas, problema.y_c[:n], ".", color="black", ms=3, label="SINAN")
    ax[0].plot(datas, sim_b["c"][:n], color="tab:blue", label="Ajuste B")
    ax[0].plot(datas, sim_t["c"][:n], color="tab:orange", ls="--", label=f"Ajuste até {FIM_TREINO.date()}")
    ax[0].axvline(FIM_TREINO, color="gray", ls=":")
    ax[0].set_ylabel("Casos notificados por dia")
    ax[0].legend()
    ax[1].plot(datas, problema.y_h[:n], ".", color="black", ms=3, label="SINAN")
    ax[1].plot(datas, sim_b["h"][:n], color="tab:blue", label="Ajuste B")
    ax[1].set_ylabel("Internações por dia")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(SAIDA_FIGURA, dpi=150)
    print(f"\nGravados: {SAIDA_PARAMETROS}, {SAIDA_CENARIOS}, {SAIDA_CURVAS}, {SAIDA_FIGURA}")


if __name__ == "__main__":
    main()
