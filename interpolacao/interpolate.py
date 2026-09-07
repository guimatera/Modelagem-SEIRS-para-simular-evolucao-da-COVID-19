import os
import glob
import calendar
import unicodedata

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from sklearn.metrics import r2_score
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

# =============================================================================
# CONFIGURACOES - ajuste aqui para diferentes conjuntos de dados
# =============================================================================

# Caminhos resolvidos a partir da pasta deste arquivo (e nao do diretorio de
# execucao), para o script e a interface funcionarem de qualquer lugar
RAIZ             = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DADOS  = os.path.join(RAIZ, "dados")                       # pasta varrida pela interface
ARQUIVO_CSV      = os.path.join(DIRETORIO_DADOS, "nascimentos_brasil.csv")  # usado ao rodar como script
SEPARADOR_CSV    = ";"                                               # separador do CSV

# Colunas lidas do cabecalho do CSV:
#   COLUNA_PERIODO -> 1a coluna (periodo)
#   COLUNA_VALOR   -> 2a coluna (valor)

# Quantidade de anos a projetar no futuro
QUANTIDADE_ANOS_PREVISAO = 10

MESES = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12
}

NOMES_MES_CURTO = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                   "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

# Modelo de tendência:
#   "exponencial" -> Decaimento Exponencial com assíntota: y = A*e^(Bx) + C
#   "polinomial"  -> Melhor polinômio de grau 1 até MAX_GRAU_POLI
MODELO_TENDENCIA = "exponencial"
MAX_GRAU_POLI    = 5  # usado apenas quando MODELO_TENDENCIA = "polinomial"

# Período opcionalmente descartado antes de interpolar e ajustar a tendência,
# no formato (ano_inicial, ano_final_exclusivo). O padrão cobre o choque da
# COVID-19: jan/2020 até dez/2022. A interface liga/desliga por checkbox.
PERIODO_COVID           = (2020.0, 2023.0)
EXCLUIR_PERIODO_PADRAO  = None  # ou PERIODO_COVID, para o script já rodar sem COVID

# =============================================================================
# 1. LEITURA E PREPARAÇÃO DOS DADOS
# =============================================================================
def listar_csvs(diretorio=DIRETORIO_DADOS):
    """Retorna a lista ordenada dos CSVs disponíveis no diretório de dados."""
    return sorted(glob.glob(os.path.join(diretorio, "*.csv")))

def ler_colunas(arquivo, separador=SEPARADOR_CSV):
    """Lê o cabeçalho do CSV: 1a coluna = período, 2a coluna = valor."""
    cabecalho = pd.read_csv(arquivo, sep=separador, nrows=0).columns
    if len(cabecalho) < 2:
        raise ValueError(
            f"'{arquivo}' precisa de ao menos 2 colunas (periodo e valor); "
            f"encontrado: {list(cabecalho)}"
        )
    return cabecalho[0], cabecalho[1]

def mes_para_frac(nome_mes):
    """Converte nome do mês (PT) para a fração do ano correspondente ao meio do mês."""
    nome = unicodedata.normalize("NFD", nome_mes.lower())
    nome = "".join(c for c in nome if unicodedata.category(c) != "Mn")
    m = MESES.get(nome, None)
    if m is None:
        return None
    return (m - 0.5) / 12.0  # meio do mês como fração do ano

def carregar_dados_mensais(arquivo, separador, col_periodo, col_valor):
    """
    Lê o CSV e extrai as linhas MENSAIS (ex: 'Janeiro/2015').
    Retorna arrays numpy (x_frac, valores) onde x_frac é o ano fracionário.
    """
    df = pd.read_csv(arquivo, sep=separador, dtype=str)
    df[col_periodo] = df[col_periodo].str.strip()
    df[col_valor]   = df[col_valor].str.replace(".", "", regex=False).str.replace(",", ".").str.strip()

    x_list, y_list = [], []
    for _, row in df.iterrows():
        periodo = row[col_periodo]
        valor_str = row[col_valor]
        # Esperado: "Mês/AAAA"
        if "/" not in periodo:
            continue
        partes = periodo.split("/")
        if len(partes) != 2:
            continue
        nome_mes_raw, ano_str = partes[0].strip(), partes[1].strip()
        if not ano_str.isdigit():
            continue
        frac = mes_para_frac(nome_mes_raw)
        if frac is None:
            continue
        try:
            val = float(valor_str)
        except ValueError:
            continue
        x_list.append(int(ano_str) + frac)
        y_list.append(val)

    return np.array(x_list), np.array(y_list)

def carregar_dados_anuais(arquivo, separador, col_periodo, col_valor):
    df = pd.read_csv(arquivo, sep=separador, dtype=str)
    df[col_periodo] = df[col_periodo].str.strip()
    df[col_valor]   = df[col_valor].str.replace(".", "", regex=False).str.replace(",", ".").str.strip()

    mask = df[col_periodo].str.fullmatch(r"\d{4}")
    df_anual = df[mask].copy()
    df_anual[col_valor] = pd.to_numeric(df_anual[col_valor], errors="coerce")
    df_anual = df_anual.dropna(subset=[col_valor])

    anos = df_anual[col_periodo].astype(int).values
    vals = df_anual[col_valor].astype(float).values
    return anos, vals


# =============================================================================
# 2. MODELOS DE TENDÊNCIA
# =============================================================================

def ajustar_exponencial(x, y):
    """Ajuste via curve_fit: y = A*e^(Bx) + C, com B<0 e C>0."""
    def _func(xv, A, B, C):
        return A * np.exp(B * xv) + C

    A0 = float(y[0] - y[-1])
    B0 = -0.03
    C0 = float(y[-1])
    bounds = ([-np.inf, -1.0, 0.0], [np.inf, 0.0, np.inf])
    try:
        popt, _ = curve_fit(_func, x, y, p0=[A0, B0, C0],
                            bounds=bounds, maxfev=30000)
        return popt, _func
    except RuntimeError:
        # Fallback: log-linear simples
        coef  = np.polyfit(x, np.log(y), 1)
        B_fb  = coef[0]
        A_fb  = float(y[0])
        C_fb  = 0.0
        return (A_fb, B_fb, C_fb), _func

def ajustar_polinomial(x, y, max_grau):
    """Escolhe o melhor polinômio (grau 1 a max_grau) por R²."""
    melhor_r2, melhor_grau, melhor_mod = -np.inf, 1, None
    for grau in range(1, max_grau + 1):
        coef  = np.polyfit(x, y, grau)
        mod   = np.poly1d(coef)
        r2    = r2_score(y, mod(x))
        if r2 > melhor_r2:
            melhor_r2, melhor_grau, melhor_mod = r2, grau, mod
    return melhor_mod, melhor_grau, melhor_r2


# =============================================================================
# 3. ANÁLISE COMPLETA DE UM CSV
# =============================================================================

def analisar_csv(arquivo,
                 separador=SEPARADOR_CSV,
                 anos_previsao=QUANTIDADE_ANOS_PREVISAO,
                 modelo=MODELO_TENDENCIA,
                 max_grau=MAX_GRAU_POLI,
                 excluir_periodo=EXCLUIR_PERIODO_PADRAO):
    """
    Interpolação (CubicSpline) + tendência + projeção futura de um CSV.
    Retorna um dicionário com os dados, modelos e métricas do arquivo.
    """
    col_periodo, col_valor = ler_colunas(arquivo, separador)

    # Dados mensais (para interpolação e scatter)
    x_mensal, y_mensal = carregar_dados_mensais(arquivo, separador, col_periodo, col_valor)
    if len(x_mensal) < 4:
        raise ValueError(f"'{arquivo}': poucos pontos mensais validos ({len(x_mensal)}).")

    # Descarta o período excluído (ex.: choque da COVID-19). Os pontos saem da
    # spline e do ajuste de tendência, mas continuam guardados para o gráfico.
    x_excluido = np.array([])
    y_excluido = np.array([])
    if excluir_periodo:
        ini, fim   = excluir_periodo
        dentro     = (x_mensal >= ini) & (x_mensal < fim)
        x_excluido, y_excluido = x_mensal[dentro], y_mensal[dentro]
        x_mensal,   y_mensal   = x_mensal[~dentro], y_mensal[~dentro]
        if len(x_mensal) < 4:
            raise ValueError(
                f"'{arquivo}': sobraram poucos pontos ({len(x_mensal)}) "
                f"apos excluir {ini:.0f}-{fim:.0f}."
            )

    # Dados anuais (ano-base e período coberto)
    anos_reais, y_anual = carregar_dados_anuais(arquivo, separador, col_periodo, col_valor)

    if excluir_periodo and len(anos_reais):
        ini, fim   = excluir_periodo
        fora_anual = (anos_reais < int(np.floor(ini))) | (anos_reais >= int(np.ceil(fim)))
        anos_reais = anos_reais[fora_anual]
        y_anual    = y_anual[fora_anual]

    # Remove anos incompletos (ex.: o ano corrente, ainda em andamento) do conjunto
    # anual. O limiar vem da MEDIANA, e nao do minimo: o ano parcial costuma ser o
    # proprio minimo da serie, e comparar com ele nunca descartava nada.
    if len(y_anual) > 0:
        limiar_completo = np.median(y_anual) * 0.6
        mask_completo   = y_anual >= limiar_completo
        anos_reais      = anos_reais[mask_completo]
        y_anual         = y_anual[mask_completo]

    # Normaliza eixo x mensal (t=0 no primeiro mês do primeiro ano)
    ano_base      = int(anos_reais[0]) if len(anos_reais) else int(np.floor(x_mensal.min()))
    x_mensal_norm = x_mensal - ano_base  # fração normalizada (0.0, 0.08, 0.17 ...)

    # Spline sobre dados mensais (granularidade maior, curva mais precisa)
    # Ordena por x para garantir que CubicSpline receba pontos em ordem crescente
    ordem     = np.argsort(x_mensal_norm)
    x         = x_mensal_norm[ordem]
    y         = y_mensal[ordem]
    spline    = CubicSpline(x, y)
    r2_spline = r2_score(y_mensal, spline(x_mensal_norm))

    # Projeção futura em passos mensais (1/12 de ano)
    x_futuros = np.arange(x[-1] + 1/12, x[-1] + 1/12 + anos_previsao, 1/12)

    # Tendência ajustada sobre dados MENSAIS (acompanha a granularidade do scatter)
    if modelo == "exponencial":
        (A_fit, B_fit, C_fit), _func_exp = ajustar_exponencial(x, y)

        r2_tend = r2_score(y, _func_exp(x, A_fit, B_fit, C_fit))

        # Projeção usa os parâmetros do próprio ajuste (sem ancoragem)
        def modelo_tendencia(xv):
            return _func_exp(xv, A_fit, B_fit, C_fit)

        rotulo_modelo = "Decaimento Exponencial"
        descricao = [
            f"Equacao: y = {A_fit:.2e} * e^({B_fit:.5f} * (Ano - {ano_base})) + {C_fit:.2e}",
            f"Assintota (C) = {C_fit:,.0f}".replace(",", "."),
        ]

    elif modelo == "polinomial":
        mod_poly, grau_poly, r2_tend = ajustar_polinomial(x, y, max_grau)

        def modelo_tendencia(xv):
            return mod_poly(xv)

        rotulo_modelo = f"Polinomio Grau {grau_poly}"
        descricao = [f"Equacao: {mod_poly}"]

    else:
        raise ValueError(
            f"MODELO_TENDENCIA invalido: '{modelo}'. Use 'exponencial' ou 'polinomial'."
        )

    estimativa_futura = modelo_tendencia(x_futuros)

    if modelo == "exponencial":
        r2_exponential_regression = r2_tend
    else:
        (A_exp, B_exp, C_exp), _func_exp_ref = ajustar_exponencial(x, y)
        r2_exponential_regression = r2_score(y, _func_exp_ref(x, A_exp, B_exp, C_exp))

    return {
        "arquivo":           arquivo,
        "nome":              os.path.splitext(os.path.basename(arquivo))[0],
        "col_periodo":       col_periodo,
        "col_valor":         col_valor,
        "x_mensal":          x_mensal,
        "y_mensal":          y_mensal,
        "x_excluido":        x_excluido,
        "y_excluido":        y_excluido,
        "excluir_periodo":   excluir_periodo,
        "anos_reais":        anos_reais,
        "y_anual":           y_anual,
        "ano_base":          ano_base,
        "x":                 x,
        "y":                 y,
        "spline":            spline,
        "r2_spline":         r2_spline,
        "modelo":            modelo,
        "rotulo_modelo":     rotulo_modelo,
        "descricao_modelo":  descricao,
        "r2_tendencia":      r2_tend,
        "r2_exponential_regression": r2_exponential_regression,
        "modelo_tendencia":  modelo_tendencia,
        "anos_previsao":     anos_previsao,
        "x_futuros":         x_futuros,
        "estimativa_futura": estimativa_futura,
    }

def ano_mes(x_absoluto):
    """
    Converte o ano fracionário (2015.04 ...) no par (ano, mes).
    Usa piso, e não arredondamento: os pontos caem no MEIO do mês
    ((m-0.5)/12), e round(m-0.5) arredondava para o par mais próximo,
    errando o mês em metade dos casos.
    """
    ano = int(np.floor(x_absoluto))
    mes = int(np.floor((x_absoluto - ano) * 12)) + 1
    return ano, max(1, min(12, mes))

def frac_para_mes_ano(frac_norm, ano_base):
    """Converte o x normalizado de volta para o rótulo 'Mes/AAAA'."""
    ano, mes = ano_mes(frac_norm + ano_base)
    return f"{NOMES_MES_CURTO[mes-1]}/{ano}"

def formata_numero(valor, casas=0):
    """Formata no padrão brasileiro: 131.912 / 4.333,5."""
    return (f"{valor:,.{casas}f}"
            .replace(",", "\x00").replace(".", ",").replace("\x00", "."))

def resumo_texto(res, incluir_previsao=True):
    """Monta o resumo textual da análise (mesmo conteúdo impresso pelo script)."""
    linhas = [
        f"Dados carregados de '{res['arquivo']}'",
        f"Colunas: '{res['col_periodo']}' (periodo) | '{res['col_valor']}' (valor)",
    ]
    if len(res["anos_reais"]):
        linhas.append(
            f"Anos: {res['anos_reais'][0]} - {res['anos_reais'][-1]}  |  "
            f"Pontos mensais: {len(res['x_mensal'])}  |  Pontos anuais: {len(res['y_anual'])}"
        )
    if res.get("excluir_periodo"):
        ini, fim = res["excluir_periodo"]
        linhas.append(
            f"Periodo excluido: {frac_para_mes_ano(ini, 0)} - "
            f"{frac_para_mes_ano(fim - 1/12, 0)}  ({len(res['y_excluido'])} meses fora do ajuste)"
        )
    linhas += [
        "",
        f"[Interpolacao] CubicSpline (mensal)  |  R2 = {res['r2_spline']:.4f}",
        "",
        f"[Tendencia] {res['rotulo_modelo']}  |  R2 (fit) = {res['r2_tendencia']:.4f}",
    ]
    linhas += [f"  {d}" for d in res["descricao_modelo"]]

    if incluir_previsao:
        linhas += [
            "",
            f"[Previsao] Estimativa para os proximos {res['anos_previsao']} anos (mensal):",
        ]
        for xf, val in zip(res["x_futuros"], res["estimativa_futura"]):
            linhas.append(
                f"  {frac_para_mes_ano(xf, res['ano_base'])}: {val:,.0f}".replace(",", ".")
            )

    return "\n".join(linhas)


# =============================================================================
# 4. PLOT (resolução diária, dados mensais no scatter)
# =============================================================================

def formata_tooltip(sel):
    """
    Tooltip interativo: clique para ver Mês/Ano, o total do mês e a média
    diária. Os pontos do gráfico são totais MENSAIS, então a média diária
    divide pelos dias daquele mês (28 a 31), e não por 365.
    """
    x_real = sel.target[0]
    y_val  = sel.target[1]
    ano, mes = ano_mes(x_real)
    dias     = calendar.monthrange(ano, mes)[1]  # ja considera ano bissexto
    sel.annotation.set_text(
        f"{NOMES_MES_CURTO[mes-1]}/{ano}\n"
        f"Total do mes: {formata_numero(y_val)}\n"
        f"Media/dia ({dias} dias): {formata_numero(y_val / dias, 1)}"
    )

def montar_figura(res, fig=None, titulo=None, label_y=None, figsize=(14, 6)):
    """
    Desenha a análise (scatter mensal + spline + projeção) em uma figura.

    fig=None cria a figura via pyplot (uso em script); passando uma Figure já
    existente (ex.: embutida na interface), desenha nela sem usar o pyplot.
    Retorna (fig, ax, cursor) - guarde o cursor para o tooltip seguir ativo.
    """
    ano_base = res["ano_base"]
    x        = res["x"]

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    # Intervalo de x para a curva de interpolação (resolução diária)
    x_plot = np.arange(x.min(), x.max(), 1/365)
    y_plot = res["spline"](x_plot)

    # Dentro do período excluído a spline não tem dado nenhum para interpolar e
    # dispara (chega a superar o pico real). NaN interrompe a linha ali.
    if res.get("excluir_periodo"):
        ini, fim = res["excluir_periodo"]
        y_plot = np.where(((x_plot + ano_base) >= ini) & ((x_plot + ano_base) < fim),
                          np.nan, y_plot)

    # Intervalo de x para projeção futura (resolução diária)
    x_fut_plot = np.arange(x[-1], res["x_futuros"][-1] + 1/365, 1/365)
    y_fut_plot = res["modelo_tendencia"](x_fut_plot)

    # Scatter mensal dos dados originais
    ax.scatter(res["x_mensal"], res["y_mensal"],
               color="crimson", zorder=5, s=18, alpha=0.7, label="Dados Mensais")

    # Pontos descartados: aparecem no grafico, mas ficam fora de todo o ajuste
    if len(res.get("x_excluido", [])):
        ini, fim = res["excluir_periodo"]
        ax.scatter(res["x_excluido"], res["y_excluido"],
                   facecolors="none", edgecolors="gray", zorder=4, s=22,
                   label=f"Excluidos ({frac_para_mes_ano(ini, 0)} - "
                         f"{frac_para_mes_ano(fim - 1/12, 0)})")
        ax.axvspan(ini, fim, color="gray", alpha=0.08, zorder=0)

    # Curva de interpolação CubicSpline
    ax.plot(x_plot + ano_base, y_plot,
            color="royalblue", lw=1.5, label="Interpolacao CubicSpline (mensal)")

    # Scatter e linha da projeção mensal futura
    ax.scatter(res["x_futuros"] + ano_base, res["estimativa_futura"],
               color="purple", zorder=5, marker="x", s=30, alpha=0.8,
               label="Estimativa Mensal Futura")
    ax.plot(x_fut_plot + ano_base, y_fut_plot,
            color="purple", lw=2, linestyle="--",
            label=f"Tendência ({res['modelo'].capitalize()})")

    titulo_padrao = f"Interpolação e Tendência - {res['col_valor']}"
    if res.get("excluir_periodo"):
        titulo_padrao += " (sem o período excluído)"
    ax.set_title(titulo or titulo_padrao, fontsize=14, fontweight="bold")
    ax.set_xlabel("Ano")
    ax.set_ylabel(label_y or res["col_valor"])
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    cursor = mplcursors.cursor(ax, hover=False)
    cursor.connect("add", formata_tooltip)

    return fig, ax, cursor


# =============================================================================
# 5. EXECUÇÃO COMO SCRIPT (a interface esta em interpolate_gui.py)
# =============================================================================

def main():
    res = analisar_csv(
        ARQUIVO_CSV,
        separador=SEPARADOR_CSV,
        anos_previsao=QUANTIDADE_ANOS_PREVISAO,
        modelo=MODELO_TENDENCIA,
        max_grau=MAX_GRAU_POLI,
    )
    print(resumo_texto(res))
    montar_figura(res)
    plt.show()


if __name__ == "__main__":
    main()
