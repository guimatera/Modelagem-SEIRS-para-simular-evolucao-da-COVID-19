import os
import glob
import calendar
import warnings
import unicodedata

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from sklearn.metrics import r2_score
from scipy.interpolate import CubicSpline
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# =============================================================================
# CONFIGURACOES - ajuste aqui para diferentes conjuntos de dados
# =============================================================================
# Sao dois metodos, um para cada metade do trabalho:
#   INTERPOLACAO  -> CubicSpline sobre os pontos mensais observados
#   EXTRAPOLACAO  -> Holt-Winters (suavizacao exponencial tripla)

# Caminhos resolvidos a partir da pasta deste arquivo (e nao do diretorio de
# execucao), para o script e a interface funcionarem de qualquer lugar
RAIZ             = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DADOS  = os.path.join(RAIZ, "dados")                       # pasta varrida pela interface
ARQUIVO_CSV      = os.path.join(DIRETORIO_DADOS, "nascimentos_brasil.csv")  # usado ao rodar como script
SEPARADOR_CSV    = ";"                                               # separador do CSV

# Colunas lidas do cabecalho do CSV: 1a = periodo ("Marco/2015"), 2a = valor

# Quantidade de anos a projetar no futuro
QUANTIDADE_ANOS_PREVISAO = 10

MESES = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12
}

NOMES_MES_CURTO = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                   "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

# --- Parâmetros do Holt-Winters ---------------------------------------------
# Tamanho do ciclo sazonal, em observações: dados mensais, ciclo anual -> 12.
PERIODO_SAZONAL = 12

# Como a sazonalidade entra no modelo:
#   "multiplicativa" -> y = (nivel + tendencia) * sazonal[mes]
#                       oscilação PROPORCIONAL ao nível (encolhe junto quando a
#                       série cai) - é o caso de nascimentos/óbitos
#   "aditiva"        -> y = (nivel + tendencia) + sazonal[mes]
#                       oscilação de amplitude aproximadamente constante
SAZONALIDADE_HW = "multiplicativa"

# Amortecimento da tendência (phi < 1): a inclinação vai perdendo força ao longo
# do horizonte em vez de seguir reta indefinidamente. DESLIGADO por padrão: em
# 120 passos à frente ele achata a projeção inteira (nos óbitos, a extrapolação
# saía numa linha plana, sem a subida de ~2.100/ano que os dados mostram).
# Ligue apenas para uma projeção deliberadamente conservadora.
AMORTECER_TENDENCIA_HW = False

# Período opcionalmente descartado antes de interpolar e extrapolar, no formato
# (ano_inicial, ano_final_exclusivo). O padrão cobre o choque da COVID-19:
# jan/2020 até dez/2022. A interface liga/desliga por checkbox.
PERIODO_COVID           = (2020.0, 2023.0)
EXCLUIR_PERIODO_PADRAO  = None  # ou PERIODO_COVID, para o script já rodar sem COVID


# =============================================================================
# 1. LEITURA DOS DADOS
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

def _numero_do_mes(nome):
    """'Março' -> 3. Devolve None se não for um nome de mês em português."""
    sem_acento = unicodedata.normalize("NFD", str(nome).strip().lower())
    sem_acento = "".join(c for c in sem_acento if unicodedata.category(c) != "Mn")
    return MESES.get(sem_acento)

def carregar_dados_mensais(arquivo, separador, col_periodo, col_valor):
    """
    Lê o CSV e devolve (x, y) das linhas MENSAIS ("Janeiro/2015"), ignorando as
    demais (totais anuais, rodapé). x é o ano fracionário com o ponto no MEIO do
    mês: 2015 + (mes - 0.5)/12.
    """
    df      = pd.read_csv(arquivo, sep=separador, dtype=str)
    partes  = df[col_periodo].str.strip().str.extract(r"^([^/]+)/(\d{4})$")
    meses   = partes[0].map(_numero_do_mes)
    valores = pd.to_numeric(
        df[col_valor].str.replace(".", "", regex=False).str.replace(",", ".").str.strip(),
        errors="coerce")

    ok = meses.notna() & valores.notna()
    x  = partes[1][ok].astype(int) + (meses[ok].astype(float) - 0.5) / 12.0
    return x.to_numpy(dtype=float), valores[ok].to_numpy(dtype=float)


# =============================================================================
# 2. EXTRAPOLACAO - HOLT-WINTERS (SUAVIZACAO EXPONENCIAL TRIPLA)
# =============================================================================
# Em vez de forcar a serie a caber numa formula fechada, o metodo mantem tres
# componentes atualizados a cada mes observado - NIVEL (l), TENDENCIA (b) e
# SAZONALIDADE (s) -, cada um uma media ponderada entre "o que o dado novo diz"
# e "o que a serie vinha dizendo", com pesos alpha, beta e gamma que dao mais
# peso ao passado recente. A previsao h meses a frente recompoe as tres partes:
#   multiplicativa: y(t+h) = (l + (phi + phi^2 + ... + phi^h) * b) * s[mes]
#   aditiva:        y(t+h) = (l + (phi + phi^2 + ... + phi^h) * b) + s[mes]
# phi < 1 e o amortecimento: a inclinacao perde forca ao longo do horizonte, em
# vez de seguir em linha reta pelos 10 anos projetados.

def _completar_serie_mensal(x, y, periodo, multiplicativa):
    """
    Devolve (grade, serie, observado) com a série mensal COMPLETA e igualmente
    espaçada que o Holt-Winters exige - ele avança de mês em mês e identifica a
    posição no ciclo pela contagem, então um buraco no meio desalinha toda a
    sazonalidade dali para frente.

    O buraco aparece quando um período é excluído (o padrão é o choque da
    COVID-19: 36 meses fora, com 60 antes e ~39 depois). Descartar o trecho
    anterior jogaria fora metade da história e emendar os dois deslocaria os
    meses, então os ausentes são RECONSTRUÍDOS (e marcados em `observado`, para
    ficarem fora do R²): nível por interpolação linear entre as bordas, vezes o
    fator típico daquele mês medido nos dados reais. Sem esse segundo passo o
    buraco entraria como 3 ciclos "chapados" e o modelo leria a sazonalidade
    como menor do que ela é.
    """
    idx   = np.rint(np.asarray(x, dtype=float) * 12 - 0.5).astype(int)  # mes 0 = jan do ano-base
    ordem = np.argsort(idx)
    idx   = idx[ordem]

    grade     = np.arange(idx[0], idx[-1] + 1)
    serie     = np.interp(grade, idx, np.asarray(y, dtype=float)[ordem])
    observado = np.isin(grade, idx)

    faltantes = ~observado
    if faltantes.any():
        nivel  = (pd.Series(serie).rolling(periodo, center=True, min_periods=1)
                  .mean().to_numpy())
        desvio = serie / nivel if multiplicativa else serie - nivel
        meses  = grade % periodo
        tipico = (pd.Series(desvio[observado]).groupby(meses[observado]).median()
                  .reindex(range(periodo)).fillna(1.0 if multiplicativa else 0.0)
                  .to_numpy())[meses[faltantes]]
        serie[faltantes] = (nivel[faltantes] * tipico if multiplicativa
                            else nivel[faltantes] + tipico)

    return grade, serie, observado

def _estado_inicial(serie, observado, grade, periodo, multiplicativa):
    """
    Estado inicial (nível, tendência, sazonalidade) medido direto nos meses
    OBSERVADOS, para o Holt-Winters partir da tendência que aparece no gráfico.

    Sem isso (`initialization_method="estimated"`), quem escolhe a inclinação
    inicial é o otimizador - e ele minimiza o erro de UM passo à frente, onde a
    tendência quase não pesa. Nos óbitos isso zerava a inclinação: o ajuste
    ficava ótimo mês a mês e a projeção de 10 anos saía plana, ignorando a
    subida clara da série. Aqui a inclinação vem de uma reta de mínimos
    quadrados sobre os pontos REAIS (os meses reconstruídos ficam de fora, para
    a ponte do período excluído não ditar a tendência) e a sazonalidade, da
    razão (ou diferença) mediana de cada mês em relação a essa reta.
    """
    t     = np.arange(len(serie), dtype=float)
    coef  = np.polyfit(t[observado], serie[observado], 1)   # [inclinacao, nivel em t=0]
    reta  = np.polyval(coef, t)
    meses = grade % periodo

    neutro  = 1.0 if multiplicativa else 0.0
    desvio  = serie / reta if multiplicativa else serie - reta
    fatores = np.array([np.median(desvio[observado & (meses == m)])
                        if (observado & (meses == m)).any() else neutro
                        for m in range(periodo)])
    # normaliza para a sazonalidade nao competir com o nivel
    fatores = fatores / fatores.mean() if multiplicativa else fatores - fatores.mean()

    # o statsmodels espera a sazonalidade dos `periodo` primeiros meses da serie
    return float(coef[1]), float(coef[0]), fatores[(int(grade[0]) + np.arange(periodo)) % periodo]

def ajustar_holt_winters(x, y, passos_futuros,
                         periodo=PERIODO_SAZONAL,
                         sazonalidade=SAZONALIDADE_HW,
                         amortecer=AMORTECER_TENDENCIA_HW):
    """
    Ajusta o Holt-Winters sobre a série mensal e devolve (prever, info):

      prever(xv) -> valor estimado em qualquer x normalizado. Dentro da amostra
                    são os valores ajustados (previsão de um passo à frente);
                    fora, a previsão. Entre dois meses interpola linearmente, só
                    para a curva do gráfico (resolução diária) ficar contínua.
      info       -> alpha/beta/gamma/phi, sazonalidade por mês e quantos meses
                    tiveram de ser reconstruídos.

    alpha, beta, gamma e phi NÃO são escolhidos à mão: o statsmodels estima os
    quatro numericamente, minimizando o erro dentro da amostra. Já o estado
    INICIAL é ancorado nos dados observados (ver `_estado_inicial`), e não
    deixado a cargo do otimizador - é o que faz a projeção seguir a tendência
    da série em vez de sair plana.
    """
    if sazonalidade not in ("multiplicativa", "aditiva"):
        raise ValueError(f"SAZONALIDADE_HW invalida: '{sazonalidade}'. "
                         f"Use 'multiplicativa' ou 'aditiva'.")

    # A sazonalidade multiplicativa divide pelo nivel: exige serie positiva.
    mult  = sazonalidade == "multiplicativa" and float(np.min(y)) > 0
    grade, serie, observado = _completar_serie_mensal(x, y, periodo, mult)
    if len(serie) < 2 * periodo:
        raise ValueError(
            f"Holt-Winters precisa de ao menos {2*periodo} meses para estimar a "
            f"sazonalidade; a serie tem {len(serie)}."
        )

    nivel_0, tendencia_0, sazonal_0 = _estado_inicial(serie, observado, grade,
                                                      periodo, mult)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # avisos de convergencia/escala do otimizador
        ajuste = ExponentialSmoothing(
            serie,
            trend="add",
            damped_trend=bool(amortecer),
            seasonal="mul" if mult else "add",
            seasonal_periods=periodo,
            initialization_method="known",
            initial_level=nivel_0,
            initial_trend=tendencia_0,
            initial_seasonal=sazonal_0,
        ).fit(optimized=True)

    # Horizonte com folga: o grafico vai ate o ultimo x futuro e o resumo precisa
    # de um ciclo inteiro para descrever a sazonalidade projetada.
    previsao = np.asarray(ajuste.forecast(int(max(passos_futuros, periodo)) + periodo),
                          dtype=float)
    curva    = np.concatenate([np.asarray(ajuste.fittedvalues, dtype=float), previsao])
    t0       = int(grade[0])

    def prever(xv):
        return np.interp(np.asarray(xv, dtype=float) * 12 - 0.5 - t0,
                         np.arange(len(curva), dtype=float), curva)

    # Sazonalidade de cada mes: tirando dos 12 primeiros meses previstos a parte
    # nivel + tendencia amortecida sobra exatamente o componente sazonal.
    par       = ajuste.params
    phi       = float(par.get("damping_trend", 1.0)) if amortecer else 1.0
    phi       = phi if np.isfinite(phi) else 1.0
    nivel     = float(np.asarray(ajuste.level)[-1])
    tendencia = float(np.asarray(ajuste.trend)[-1])
    h         = np.arange(1, periodo + 1)
    base      = nivel + np.cumsum(phi ** h) * tendencia
    fatores   = previsao[:periodo] / base if mult else previsao[:periodo] - base
    meses     = ((int(grade[-1]) + h) % periodo) + 1   # 1 = janeiro

    info = {
        "multiplicativa":  mult,
        "alpha":           float(par.get("smoothing_level", np.nan)),
        "beta":            float(par.get("smoothing_trend", np.nan)),
        "gamma":           float(par.get("smoothing_seasonal", np.nan)),
        "phi":             phi if amortecer else None,
        "nivel_final":       nivel,
        "tendencia_final":   tendencia,
        "tendencia_ancora":  tendencia_0,
        "mes_pico":        int(meses[int(np.argmax(fatores))]),
        "mes_vale":        int(meses[int(np.argmin(fatores))]),
        "amplitude":       float((np.max(fatores) - np.min(fatores)) / 2.0),
        "meses_totais":    int(len(serie)),
        "meses_reconstruidos": int(np.count_nonzero(~observado)),
    }
    return prever, info


# =============================================================================
# 3. ANÁLISE COMPLETA DE UM CSV
# =============================================================================

def analisar_csv(arquivo,
                 separador=SEPARADOR_CSV,
                 anos_previsao=QUANTIDADE_ANOS_PREVISAO,
                 excluir_periodo=EXCLUIR_PERIODO_PADRAO):
    """
    Interpolação (CubicSpline) + extrapolação (Holt-Winters) de um CSV.
    Retorna um dicionário com os dados, os modelos e as métricas do arquivo.
    """
    col_periodo, col_valor = ler_colunas(arquivo, separador)
    x_mensal, y_mensal = carregar_dados_mensais(arquivo, separador, col_periodo, col_valor)

    # Descarta o período excluído (ex.: choque da COVID-19). Os pontos saem da
    # spline e do ajuste, mas continuam guardados para aparecer no gráfico.
    x_excluido = y_excluido = np.array([])
    if excluir_periodo:
        ini, fim = excluir_periodo
        dentro   = (x_mensal >= ini) & (x_mensal < fim)
        x_excluido, y_excluido = x_mensal[dentro], y_mensal[dentro]
        x_mensal,   y_mensal   = x_mensal[~dentro], y_mensal[~dentro]

    if len(x_mensal) < 4:
        raise ValueError(f"'{arquivo}': poucos pontos mensais validos ({len(x_mensal)}).")

    # Eixo x normalizado: t=0 em janeiro do primeiro ano com dado, e ordenado
    # (CubicSpline exige x crescente).
    ano_base = int(np.floor(x_mensal.min()))
    ordem    = np.argsort(x_mensal)
    x        = x_mensal[ordem] - ano_base
    y        = y_mensal[ordem]

    # --- Interpolacao: spline cubica passando por todos os pontos mensais ----
    spline    = CubicSpline(x, y)
    r2_spline = r2_score(y, spline(x))

    # --- Extrapolacao: Holt-Winters, em passos mensais (1/12 de ano) ---------
    x_futuros = np.arange(x[-1] + 1/12, x[-1] + 1/12 + anos_previsao, 1/12)
    modelo_tendencia, hw = ajustar_holt_winters(
        x, y, passos_futuros=int(round(anos_previsao * PERIODO_SAZONAL)))

    # R2 medido SO nos meses realmente observados (os reconstruidos para fechar
    # o buraco do periodo excluido nao entram na conta).
    r2_tend = r2_score(y, modelo_tendencia(x))

    rotulo = ("Holt-Winters ("
              + ("multiplicativa" if hw["multiplicativa"] else "aditiva")
              + (", amortecido)" if hw["phi"] is not None else ")"))

    suavizacao = (f"Suavizacao: alpha (nivel) = {hw['alpha']:.4f}  |  "
                  f"beta (tendencia) = {hw['beta']:.4f}  |  "
                  f"gamma (sazonalidade) = {hw['gamma']:.4f}")
    if hw["phi"] is not None:
        suavizacao += f"  |  phi (amortecimento) = {hw['phi']:.4f}"

    amplitude = (f"{hw['amplitude']*100:.1f}% do nivel" if hw["multiplicativa"]
                 else formata_numero(hw["amplitude"]) + " (unidades)")
    descricao = [
        f"Equacao: y(t+h) = (nivel + "
        + ("(phi + phi^2 + ... + phi^h)" if hw["phi"] is not None else "h")
        + " * tendencia) " + ("*" if hw["multiplicativa"] else "+") + " sazonal[mes]",
        suavizacao,
        f"Estado final: nivel = {formata_numero(hw['nivel_final'])}  |  "
        f"tendencia = {formata_numero(hw['tendencia_final'])}/mes "
        f"({formata_numero(hw['tendencia_final']*12)}/ano)",
        f"Tendencia ancorada nos dados observados (minimos quadrados): "
        f"{formata_numero(hw['tendencia_ancora']*12)}/ano",
        f"Ciclo sazonal de {PERIODO_SAZONAL} meses  |  "
        f"Pico em {NOMES_MES_CURTO[hw['mes_pico']-1]}  |  "
        f"Vale em {NOMES_MES_CURTO[hw['mes_vale']-1]}  |  "
        f"Amplitude sazonal = +-{amplitude}",
    ]
    if hw["meses_reconstruidos"]:
        descricao.append(
            f"Serie mensal: {hw['meses_totais']} meses, sendo "
            f"{hw['meses_reconstruidos']} reconstruidos (nivel interpolado x "
            f"sazonalidade tipica) so para manter o ciclo alinhado - eles nao "
            f"entram no R2"
        )

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
        "ano_base":          ano_base,
        "x":                 x,
        "y":                 y,
        "spline":            spline,
        "r2_spline":         r2_spline,
        "rotulo_modelo":     rotulo,
        "descricao_modelo":  descricao,
        "r2_tendencia":      r2_tend,
        "modelo_tendencia":  modelo_tendencia,
        "anos_previsao":     anos_previsao,
        "x_futuros":         x_futuros,
        "estimativa_futura": modelo_tendencia(x_futuros),
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
    ano_base = res["ano_base"]
    linhas = [
        f"Dados carregados de '{res['arquivo']}'",
        f"Colunas: '{res['col_periodo']}' (periodo) | '{res['col_valor']}' (valor)",
        f"Periodo: {frac_para_mes_ano(res['x'][0], ano_base)} - "
        f"{frac_para_mes_ano(res['x'][-1], ano_base)}  |  "
        f"Pontos mensais: {len(res['x_mensal'])}",
    ]
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
        f"[Extrapolacao] {res['rotulo_modelo']}  |  R2 (fit) = {res['r2_tendencia']:.4f}",
    ]
    linhas += [f"  {d}" for d in res["descricao_modelo"]]

    if incluir_previsao:
        linhas += [
            "",
            f"[Previsao] Estimativa para os proximos {res['anos_previsao']} anos (mensal):",
        ]
        linhas += [f"  {frac_para_mes_ano(xf, ano_base)}: {formata_numero(val)}"
                   for xf, val in zip(res["x_futuros"], res["estimativa_futura"])]

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
    x_real, y_val = sel.target[0], sel.target[1]
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

    # Interpolacao e projecao desenhadas em resolucao diaria
    x_plot = np.arange(x[0], x[-1], 1/365)
    y_plot = res["spline"](x_plot)
    x_fut  = np.arange(x[-1], res["x_futuros"][-1] + 1/365, 1/365)

    # Dentro do período excluído a spline não tem dado nenhum para interpolar e
    # dispara (chega a superar o pico real). NaN interrompe a linha ali.
    if res.get("excluir_periodo"):
        ini, fim = res["excluir_periodo"]
        y_plot = np.where(((x_plot + ano_base) >= ini) & ((x_plot + ano_base) < fim),
                          np.nan, y_plot)
        ax.scatter(res["x_excluido"], res["y_excluido"],
                   facecolors="none", edgecolors="gray", zorder=4, s=22,
                   label=f"Excluidos ({frac_para_mes_ano(ini, 0)} - "
                         f"{frac_para_mes_ano(fim - 1/12, 0)})")
        ax.axvspan(ini, fim, color="gray", alpha=0.08, zorder=0)

    ax.scatter(res["x_mensal"], res["y_mensal"],
               color="crimson", zorder=5, s=18, alpha=0.7, label="Dados Mensais")
    ax.plot(x_plot + ano_base, y_plot,
            color="royalblue", lw=1.5, label="Interpolacao CubicSpline (mensal)")
    ax.scatter(res["x_futuros"] + ano_base, res["estimativa_futura"],
               color="purple", zorder=5, marker="x", s=30, alpha=0.8,
               label="Estimativa Mensal Futura")
    ax.plot(x_fut + ano_base, res["modelo_tendencia"](x_fut),
            color="purple", lw=2, linestyle="--", label="Extrapolacao Holt-Winters")

    titulo_padrao = f"Interpolação e Extrapolação - {res['col_valor']}"
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
    res = analisar_csv(ARQUIVO_CSV,
                       separador=SEPARADOR_CSV,
                       anos_previsao=QUANTIDADE_ANOS_PREVISAO)
    print(resumo_texto(res))
    montar_figura(res)
    plt.show()


if __name__ == "__main__":
    main()
