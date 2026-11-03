import os
import glob
import calendar
import datetime
import textwrap
import warnings
import unicodedata

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import mplcursors
from sklearn.metrics import r2_score
from scipy.interpolate import PchipInterpolator
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# =============================================================================
# CONFIGURACOES - ajuste aqui para diferentes conjuntos de dados
# =============================================================================
# Sao dois metodos, um para cada metade do trabalho:
#   INTERPOLACAO  -> Histospline (PCHIP sobre o acumulado), conserva os totais mensais
#   EXTRAPOLACAO  -> Holt-Winters (suavizacao exponencial tripla)

# Caminhos resolvidos a partir da pasta deste arquivo (e nao do diretorio de
# execucao), para o script e a interface funcionarem de qualquer lugar
RAIZ             = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DADOS  = os.path.join(RAIZ, "..", "dados", "series_mensais")                 # pasta varrida pela interface
ARQUIVO_CSV      = os.path.join(DIRETORIO_DADOS, "nascimentos_mensais_brasil.csv")  # usado ao rodar como script
SEPARADOR_CSV    = ";"                                               # separador do CSV
DIRETORIO_SAIDA_DIARIA = os.path.join(RAIZ, "..", "dados", "resultados", "interpolacao")  # <nome>_diaria.csv

# Taxa diaria de mortalidade (Pi do SEIHRVS) = obitos / populacao, ambos diarios
ARQUIVO_OBITOS     = os.path.join(DIRETORIO_DADOS, "obitos_mensais_brasil.csv")
ARQUIVO_POPULACAO  = os.path.join(RAIZ, "..", "dados", "populacao",
                                  "projecao_ibge_populacao_regioes_2000_2070.csv")
LINHA_POPULACAO    = "Total"   # linha do Brasil no CSV do IBGE

# Colunas lidas do cabecalho do CSV: 1a = periodo ("Marco/2015"), 2a = valor

# Quantidade de anos a projetar no futuro
QUANTIDADE_ANOS_PREVISAO = 2

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

# Picos de surto nas séries de vacina, que não se repetem de um ano para o outro:
# set/2016 (~2,5x um mês típico na 2ª dose) e o surto de sarampo de 2018-2019,
# com campanhas e vacinação de bloqueio (ago/2019 chega a ~5x). Eles inclinam a
# tendência do Holt-Winters para baixo a ponto de a projeção ficar negativa. Não
# há opção na interface; para descartá-los ao rodar o script com um CSV de vacina,
# use EXCLUIR_PERIODO_PADRAO = PERIODOS_SURTO_VACINA + [PERIODO_COVID].
PERIODOS_SURTO_VACINA   = [(2016 + 8/12, 2016 + 9/12),   # set/2016
                           (2018.0, 2020.0)]             # surto de sarampo


# =============================================================================
# 1. LEITURA DOS DADOS
# =============================================================================

def listar_csvs(diretorio=DIRETORIO_DADOS):
    """Retorna a lista ordenada dos CSVs disponíveis no diretório de dados."""
    return sorted(glob.glob(os.path.join(diretorio, "*.csv")))

def unir_periodos(periodos):
    """
    Normaliza o período excluído: aceita None, um (ini, fim) ou uma lista deles e
    devolve a lista ordenada, com os trechos encostados ou sobrepostos fundidos.
    """
    if not periodos:
        return []
    if np.isscalar(periodos[0]):
        periodos = [periodos]
    unidos = []
    for ini, fim in sorted(periodos):
        if unidos and ini <= unidos[-1][1] + 1e-9:
            unidos[-1] = (unidos[-1][0], max(unidos[-1][1], fim))
        else:
            unidos.append((ini, fim))
    return unidos

def _dentro_dos_periodos(x, periodos):
    """Máscara dos x (ano fracionário) que caem em algum dos períodos."""
    x = np.asarray(x, dtype=float)
    dentro = np.zeros(x.shape, dtype=bool)
    for ini, fim in periodos:
        dentro |= (x >= ini) & (x < fim)
    return dentro

def descrever_periodos(periodos):
    """[(2016.67, 2016.75), (2018, 2023)] -> 'Set/2016; Jan/2018 - Dez/2022'."""
    textos = []
    for ini, fim in unir_periodos(periodos):
        # meio mês para dentro de cada borda, para o erro de float não cair no
        # mês vizinho (2016 + 8/12 pode sair 2016.66666...)
        de, ate = frac_para_mes_ano(ini + 1/24, 0), frac_para_mes_ano(fim - 1/24, 0)
        textos.append(de if de == ate else f"{de} - {ate}")
    return "; ".join(textos)

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
# 2. INTERPOLACAO - HISTOSPLINE (CONSERVA OS TOTAIS MENSAIS)
# =============================================================================
# Cada ponto e o TOTAL de um mes, e nao o valor da serie num instante. Uma spline
# que passa pelos pontos (no meio do mes) nao devolve esses totais quando
# integrada. Aqui se interpola o ACUMULADO A(x), conhecido exatamente nas bordas
# dos meses, e a curva e a derivada A'(x): a integral de cada mes e, por
# construcao, A(fim) - A(inicio) = total do mes (Boneva, Kendall & Stefanov,
# 1971). Com PCHIP o acumulado e monotono em cada mes, entao a curva tem o sinal
# do total do mes (nunca negativa numa serie de contagens).
#
# O acumulado e montado sobre DIAS de calendario (fevereiro tem 28 ou 29), entao
# a curva e uma TAXA DIARIA (unidades/dia) e o mes curto deixa de aparecer como
# queda. A soma dos valores diarios de um mes e o total do mes.

def _inicio_do_mes(indice):
    """Mês absoluto (ano*12 + mes-1) -> ano fracionário do 1º dia, em datas reais."""
    ano, mes = divmod(int(indice), 12)
    dias_ano = 366 if calendar.isleap(ano) else 365
    return ano + (datetime.date(ano, mes + 1, 1).timetuple().tm_yday - 1) / dias_ano

def _ano_para_dia(x_abs):
    """Ano fracionário (datas reais) -> dia ordinal contínuo (date.toordinal)."""
    x_abs = np.asarray(x_abs, dtype=float)
    ano   = np.floor(x_abs).astype(int)
    jan1  = np.array([datetime.date(a, 1, 1).toordinal() for a in ano.ravel()],
                     dtype=float).reshape(ano.shape)
    dias  = np.where([calendar.isleap(a) for a in ano.ravel()], 366, 365).reshape(ano.shape)
    return jan1 + (x_abs - ano) * dias

def _dia_para_ano(dia):
    """Inverso de `_ano_para_dia`."""
    dia = np.asarray(dia, dtype=float)
    ano = np.array([datetime.date.fromordinal(int(d)).year for d in np.floor(dia).ravel()]
                   ).reshape(dia.shape)
    jan1 = np.array([datetime.date(a, 1, 1).toordinal() for a in ano.ravel()],
                    dtype=float).reshape(dia.shape)
    dias = np.where([calendar.isleap(a) for a in ano.ravel()], 366, 365).reshape(dia.shape)
    return ano + (dia - jan1) / dias

def histopolar(bordas_dias, totais):
    """
    Histopolação genérica: `totais[i]` é o total do intervalo
    [bordas_dias[i], bordas_dias[i+1]) (dias ordinais, intervalos encostados).
    Devolve o acumulado A(dia) (PCHIP); a taxa diária é A'(dia) e o total de
    um dia d é A(d+1) - A(d). Serve para meses (Histospline) ou anos.
    """
    return PchipInterpolator(np.asarray(bordas_dias, dtype=float),
                             np.concatenate([[0.0], np.cumsum(np.asarray(totais, dtype=float))]))

class Histospline:
    """
    Curva que conserva os totais mensais. x em anos fracionários normalizados
    (x_absoluto - ano_base), com os pontos no meio do mês como em
    `carregar_dados_mensais`; y são os totais.

    Meses ausentes (períodos excluídos) partem a série em trechos independentes:
    não há total para acumular no buraco, e ali a curva vale NaN.

    Os totais podem incluir meses PROJETADOS (Holt-Winters, que roda sobre os
    totais mensais reais): a desagregação vem depois da previsão, e a curva
    segue contínua do último mês observado para o primeiro previsto.
    `x_ultimo_observado` marca onde termina o dado real (padrão: o último x);
    `fim_observado` é a borda final desse mês.
    """
    def __init__(self, x, y, ano_base, x_ultimo_observado=None):
        idx   = np.rint((np.asarray(x, dtype=float) + ano_base) * 12 - 0.5).astype(int)
        ordem = np.argsort(idx)
        idx, y = idx[ordem], np.asarray(y, dtype=float)[ordem]
        ultimo = (idx[-1] if x_ultimo_observado is None
                  else int(np.rint((x_ultimo_observado + ano_base) * 12 - 0.5)))
        self.ano_base      = ano_base
        self.fim_observado = _inicio_do_mes(ultimo + 1) - ano_base

        # Por dentro tudo em dias ordinais: o acumulado A(dia) e a PCHIP, e a
        # taxa diaria e A'(dia). As bordas em x (anos) ficam para o grafico.
        self.trechos = []   # (x_inicio, x_fim, acumulado PCHIP em dias)
        self.meses   = []   # (x_inicio, x_fim, total do mes)
        self.dia_fim_observado = int(round(float(_ano_para_dia(self.fim_observado + ano_base))))
        cortes = np.flatnonzero(np.diff(idx) != 1) + 1
        for i_seg, y_seg in zip(np.split(idx, cortes), np.split(y, cortes)):
            bordas = np.array([_inicio_do_mes(k) - ano_base
                               for k in range(i_seg[0], i_seg[-1] + 2)])
            dias = np.array([datetime.date(k // 12, k % 12 + 1, 1).toordinal()
                             for k in range(i_seg[0], i_seg[-1] + 2)], dtype=float)
            acumulado = histopolar(dias, y_seg)
            self.trechos.append((float(bordas[0]), float(bordas[-1]), acumulado))
            self.meses += list(zip(bordas[:-1], bordas[1:], y_seg))

    def __call__(self, xv):
        """Taxa diária (unidades/dia) em x (anos normalizados); NaN fora dos trechos."""
        xv  = np.asarray(xv, dtype=float)
        out = np.full(xv.shape, np.nan)
        for a, b, acumulado in self.trechos:
            dentro = (xv >= a) & (xv <= b)
            if dentro.any():
                out[dentro] = acumulado.derivative()(_ano_para_dia(xv[dentro] + self.ano_base))
        return out

    def integrate(self, a, b):
        """Total acumulado entre x=a e x=b (anos), pulando os buracos entre trechos."""
        total = 0.0
        for ta, tb, acumulado in self.trechos:
            ini, fim = max(a, ta), min(b, tb)
            if fim > ini:
                d_ini, d_fim = _ano_para_dia(np.array([ini, fim]) + self.ano_base)
                total += float(acumulado(d_fim) - acumulado(d_ini))
        return total

    def erro_conservacao(self):
        """Maior |∫ do mês - total| entre os meses: ~0 (erro de float)."""
        return max(abs(self.integrate(a, b) - total) for a, b, total in self.meses)

    def diario(self):
        """
        Série diária por data de calendário: valor de cada dia = A(dia+1) - A(dia)
        (o total daquele dia, que somado no mês dá o total mensal). Coluna
        `observado` separa o dado real dos meses projetados.
        """
        partes = []
        for _, _, acumulado in self.trechos:
            d0, d1 = acumulado.x[0], acumulado.x[-1]
            dias   = np.arange(d0, d1)
            partes.append(pd.DataFrame({
                "dia":       [datetime.date.fromordinal(int(d)) for d in dias],
                "valor":     acumulado(dias + 1) - acumulado(dias),
                "observado": dias < self.dia_fim_observado,
            }))
        return pd.concat(partes, ignore_index=True)

# =============================================================================
# 3. EXTRAPOLACAO - HOLT-WINTERS (SUAVIZACAO EXPONENCIAL TRIPLA)
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
# 4. ANÁLISE COMPLETA DE UM CSV
# =============================================================================

def analisar_csv(arquivo,
                 separador=SEPARADOR_CSV,
                 anos_previsao=QUANTIDADE_ANOS_PREVISAO,
                 excluir_periodo=EXCLUIR_PERIODO_PADRAO):
    """
    Interpolação (histospline) + extrapolação (Holt-Winters) de um CSV.
    Retorna um dicionário com os dados, os modelos e as métricas do arquivo.
    excluir_periodo: None, um (ini, fim) ou uma lista deles.
    """
    col_periodo, col_valor = ler_colunas(arquivo, separador)
    x_mensal, y_mensal = carregar_dados_mensais(arquivo, separador, col_periodo, col_valor)
    return analisar_serie(x_mensal, y_mensal, arquivo, col_periodo, col_valor,
                          anos_previsao, excluir_periodo)

def analisar_diferenca(arquivo_a, arquivo_b,
                       separador=SEPARADOR_CSV,
                       excluir_periodo=EXCLUIR_PERIODO_PADRAO,
                       nome=None):
    """
    Interpolação (sem extrapolação) da série DIFERENÇA (A - B), mês a mês, e a
    integral da spline sobre o período com dados (res["integral"], ver
    `integrar_spline`). Só entram os meses presentes nos dois CSVs.
    """
    per_a, col_a = ler_colunas(arquivo_a, separador)
    per_b, col_b = ler_colunas(arquivo_b, separador)
    x_a, y_a = carregar_dados_mensais(arquivo_a, separador, per_a, col_a)
    x_b, y_b = carregar_dados_mensais(arquivo_b, separador, per_b, col_b)

    # Casa os meses pelo índice inteiro (evita comparar floats)
    idx_a = np.rint(x_a * 12 - 0.5).astype(int)
    idx_b = np.rint(x_b * 12 - 0.5).astype(int)
    comuns, ia, ib = np.intersect1d(idx_a, idx_b, return_indices=True)
    if len(comuns) == 0:
        raise ValueError(f"'{arquivo_a}' e '{arquivo_b}' nao tem nenhum mes em comum.")

    col_valor = f"{col_a} - {col_b}"
    res = analisar_serie(x_a[ia], y_a[ia] - y_b[ib],
                         nome or f"{os.path.basename(arquivo_a)} - {os.path.basename(arquivo_b)}",
                         "Ano/mes", col_valor, excluir_periodo=excluir_periodo,
                         extrapolar=False)
    res["integral"] = integrar_spline(res)
    return res

def integrar_spline(res):
    """
    Integral da curva de interpolação sobre os trechos com dado, pulando os
    períodos excluídos (lá não há total para conservar).

    A curva é uma taxa diária, então ∫y dt (t em dias) sai na mesma unidade da
    série (ex.: pessoas). Os limites são as bordas reais do primeiro
    e do último mês de cada trecho, então, pela histospline, a integral é igual
    à soma dos totais mensais (vai junto para comparação).
    Devolve dict com a integral, a soma mensal e os trechos integrados.
    """
    spline, ano_base = res["spline"], res["ano_base"]
    # so o trecho observado: meses projetados nao entram na integral
    trechos = [(a, min(b, spline.fim_observado)) for a, b, _ in spline.trechos
               if a < spline.fim_observado]

    return {
        "integral":        sum(spline.integrate(a, b) for a, b in trechos),  # ∫ y dt, t em dias
        "soma_mensal":     float(np.sum(res["y"])),
        "trechos":         [(a + ano_base, b + ano_base) for a, b in trechos],
        "meses_cobertos":  sum(1 for a, _, _ in spline.meses if a < spline.fim_observado),
    }

def analisar_serie(x_mensal, y_mensal, arquivo, col_periodo, col_valor,
                   anos_previsao=QUANTIDADE_ANOS_PREVISAO,
                   excluir_periodo=EXCLUIR_PERIODO_PADRAO,
                   extrapolar=True):
    """Núcleo de analisar_csv: recebe a série mensal (x em ano fracionário, y)."""
    # Descarta os períodos excluídos (ex.: choque da COVID-19, picos de surto). Os
    # pontos saem da spline e do ajuste, mas continuam guardados para o gráfico.
    periodos   = unir_periodos(excluir_periodo)
    x_excluido = y_excluido = np.array([])
    if periodos:
        dentro = _dentro_dos_periodos(x_mensal, periodos)
        x_excluido, y_excluido = x_mensal[dentro], y_mensal[dentro]
        x_mensal,   y_mensal   = x_mensal[~dentro], y_mensal[~dentro]

    if len(x_mensal) < 4:
        raise ValueError(f"'{arquivo}': poucos pontos mensais validos ({len(x_mensal)}).")

    # Eixo x normalizado: t=0 em janeiro do primeiro ano com dado, e ordenado.
    ano_base = int(np.floor(x_mensal.min()))
    ordem    = np.argsort(x_mensal)
    x        = x_mensal[ordem] - ano_base
    y        = y_mensal[ordem]

    # --- Interpolacao: histospline, conserva o total de cada mes -------------
    spline = Histospline(x, y, ano_base)

    base = {
        "arquivo":           arquivo,
        "nome":              os.path.splitext(os.path.basename(arquivo))[0],
        "col_periodo":       col_periodo,
        "col_valor":         col_valor,
        "x_mensal":          x_mensal,
        "y_mensal":          y_mensal,
        "x_excluido":        x_excluido,
        "y_excluido":        y_excluido,
        "excluir_periodo":   periodos,          # lista de (ini, fim), vazia se nenhum
        "ano_base":          ano_base,
        "x":                 x,
        "y":                 y,
        "spline":            spline,
        "erro_conservacao":  spline.erro_conservacao(),   # ~0: confere a construcao
        "diaria":            spline.diario(),             # dia, valor, observado
    }

    # Sem extrapolacao: so a interpolacao (modelo_tendencia = None sinaliza ao
    # grafico, ao resumo e a interface que nao ha curva projetada)
    if not extrapolar:
        return {**base,
                "rotulo_modelo":     None,
                "descricao_modelo":  [],
                "r2_tendencia":      None,
                "modelo_tendencia":  None,
                "anos_previsao":     0,
                "x_futuros":         np.array([]),
                "estimativa_futura": np.array([])}

    # --- Extrapolacao: Holt-Winters, em passos mensais (1/12 de ano) ---------
    # contagem inteira de meses: np.arange com passo 1/12 as vezes inclui um a mais
    x_futuros = x[-1] + np.arange(1, int(round(anos_previsao * 12)) + 1) / 12
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

    # Ordem: Holt-Winters sobre os totais mensais REAIS (acima) e so depois a
    # desagregacao pela histospline, do passado e da projecao juntos. O
    # contrario (rodar o Holt-Winters sobre dias inventados) poria no modelo a
    # forma intra-mes suposta pela histospline, que nao vem dos dados.
    estimativa = modelo_tendencia(x_futuros)
    spline     = Histospline(np.concatenate([x, x_futuros]),
                             np.concatenate([y, estimativa]),
                             ano_base, x_ultimo_observado=x[-1])

    return {
        **base,
        "spline":            spline,
        "erro_conservacao":  spline.erro_conservacao(),
        "diaria":            spline.diario(),   # observado + previsao, por dia
        "rotulo_modelo":     rotulo,
        "descricao_modelo":  descricao,
        "r2_tendencia":      r2_tend,
        "modelo_tendencia":  modelo_tendencia,
        "anos_previsao":     anos_previsao,
        "x_futuros":         x_futuros,
        "estimativa_futura": estimativa,
    }

# --- Taxa diaria de mortalidade ---------------------------------------------
# Pi(dia) = obitos(dia) / populacao(dia), por pessoa por dia (unidade do Pi do
# SEIHRVS). Obitos: Holt-Winters nos totais mensais e histopolacao diaria do
# observado + projetado (analisar_csv). Populacao: projecao anual do IBGE,
# histopolada para dias.
# Hipoteses (os dados nao permitem medi-las):
#   - o valor anual do IBGE e a MEDIA da populacao no ano: a soma dos dias do ano
#     da valor x dias do ano. O arquivo nao informa a data de referencia;
#   - obitos de todas as causas sobre a populacao residente total: Pi igual para
#     todas as idades e compartimentos, como no modelo;
#   - a forma dentro do mes (obitos) e do ano (populacao) vem so da suavidade da
#     PCHIP no acumulado.

def ler_populacao_anual(arquivo=ARQUIVO_POPULACAO, linha=LINHA_POPULACAO):
    """CSV do Tabnet/IBGE (latin-1, ';', cabeçalho na 4ª linha) -> Series ano -> população."""
    df = pd.read_csv(arquivo, sep=";", encoding="latin-1", skiprows=3, dtype=str)
    df = df[df.iloc[:, 0].str.strip() == linha]
    if df.empty:
        raise ValueError(f"Linha '{linha}' não encontrada em '{arquivo}'.")
    serie = df.iloc[0, 1:]
    serie = serie[serie.index.str.fullmatch(r"\d{4}")]
    return pd.Series(serie.astype(float).to_numpy(), index=serie.index.astype(int), name="populacao")

def populacao_diaria(pop_anual):
    """
    Histopolação anual -> diária: cada ano entra com população x dias do ano
    (pessoa-dia), e o valor de um dia é A(d+1) - A(d), a população média
    daquele dia. Devolve Series indexada pela data.
    """
    anos = pop_anual.index.to_numpy()
    if not np.all(np.diff(anos) == 1):
        raise ValueError("A projeção anual precisa de anos consecutivos.")
    bordas = np.array([datetime.date(int(a), 1, 1).toordinal() for a in np.append(anos, anos[-1] + 1)],
                      dtype=float)
    acumulado = histopolar(bordas, pop_anual.to_numpy() * np.diff(bordas))
    dias = np.arange(bordas[0], bordas[-1])
    return pd.Series(acumulado(dias + 1) - acumulado(dias),
                     index=pd.to_datetime([datetime.date.fromordinal(int(d)) for d in dias]),
                     name="populacao")

def analisar_mortalidade(anos_previsao=QUANTIDADE_ANOS_PREVISAO,
                         excluir_periodo=EXCLUIR_PERIODO_PADRAO,
                         arquivo_obitos=ARQUIVO_OBITOS,
                         arquivo_populacao=ARQUIVO_POPULACAO):
    """
    Taxa diária de mortalidade Pi = óbitos / população. Devolve o resultado da
    análise dos óbitos (Holt-Winters, histospline) com `diaria` trocada por
    dia, valor (= Pi, por pessoa-dia), obitos, populacao e observado, e:
      mortalidade_anual -> por ano: óbitos, população média, Pi médio
      erro_populacao    -> maior |média diária do ano - valor do IBGE| (~0)
      res_obitos        -> a análise original dos óbitos
    """
    res_obitos = analisar_csv(arquivo_obitos, anos_previsao=anos_previsao,
                              excluir_periodo=excluir_periodo)
    pop_anual  = ler_populacao_anual(arquivo_populacao)
    pop        = populacao_diaria(pop_anual)

    obitos = res_obitos["diaria"]
    datas  = pd.to_datetime(obitos["dia"])
    faltando = pd.DatetimeIndex(datas).difference(pop.index)
    if len(faltando):
        raise ValueError(f"A projeção do IBGE não cobre {len(faltando)} dias "
                         f"({faltando.min().date()} a {faltando.max().date()}).")
    populacao = pop.reindex(datas).to_numpy()

    diaria = pd.DataFrame({
        "dia":       obitos["dia"],
        "valor":     obitos["valor"].to_numpy() / populacao,   # Pi, por pessoa por dia
        "obitos":    obitos["valor"].to_numpy(),
        "populacao": populacao,
        "observado": obitos["observado"].to_numpy(),
    })

    ano = datas.dt.year.to_numpy()
    g   = diaria.groupby(ano)
    anual = pd.DataFrame({
        "dias":            g.size(),
        "obitos":          g["obitos"].sum(),
        "populacao_media": g["populacao"].mean(),
        "Pi_medio":        g["valor"].mean(),
    })
    media_ano = pop.groupby(pop.index.year).mean()

    return {
        **res_obitos,
        "tipo":              "mortalidade",
        "arquivo":           "mortalidade_brasil",       # chave da janela na interface
        "nome":              "mortalidade_brasil",       # -> mortalidade_brasil_diaria.csv
        "col_valor":         "Taxa diária de mortalidade (Pi, óbitos por pessoa por dia)",
        "rotulo_modelo":     f"{res_obitos['rotulo_modelo']} dos óbitos",
        "diaria":            diaria,
        "mortalidade_anual": anual,
        "populacao_anual":   pop_anual,
        "erro_populacao":    float((media_ano - pop_anual).abs().max()),
        "arquivo_obitos":    arquivo_obitos,
        "arquivo_populacao": arquivo_populacao,
        "res_obitos":        res_obitos,
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

def descrever_integral(integ):
    """'∫ y dt = 123.456 (Jan/2021 - Ago/2026, 67 meses)'."""
    # b e a borda INICIAL do mes seguinte: meio mes para tras rotula o ultimo mes
    trechos = "; ".join(f"{frac_para_mes_ano(a + 1/24, 0)} - {frac_para_mes_ano(b - 1/24, 0)}"
                        for a, b in integ["trechos"])
    return (f"∫ y dt (t em dias) ={formata_numero(integ['integral'])}  "
            f"({trechos}; {formata_numero(integ['meses_cobertos'])} meses completos)")

def resumo_texto(res, incluir_previsao=True):
    """Monta o resumo textual da análise (mesmo conteúdo impresso pelo script)."""
    if res.get("tipo") == "mortalidade":
        return resumo_mortalidade(res)
    ano_base = res["ano_base"]
    linhas = [
        f"Dados carregados de '{res['arquivo']}'",
        f"Colunas: '{res['col_periodo']}' (periodo) | '{res['col_valor']}' (valor)",
        f"Periodo: {frac_para_mes_ano(res['x'][0], ano_base)} - "
        f"{frac_para_mes_ano(res['x'][-1], ano_base)}  |  "
        f"Pontos mensais: {len(res['x_mensal'])}",
    ]
    if res.get("excluir_periodo"):
        linhas.append(
            f"Periodo excluido: {descrever_periodos(res['excluir_periodo'])}  "
            f"({len(res['y_excluido'])} meses fora do ajuste)"
        )
    linhas += [
        "",
        f"[Interpolacao] Histospline (PCHIP no acumulado)  |  maior erro do total "
        f"mensal = {res['erro_conservacao']:.2e}",
    ]
    if res.get("integral"):
        integ = res["integral"]
        linhas += [
            "",
            f"[Integral] {descrever_integral(integ)}",
            f"  Soma dos totais mensais (comparacao): {formata_numero(integ['soma_mensal'])}",
        ]
    if res["modelo_tendencia"] is not None:
        linhas += [
            "",
            f"[Extrapolacao] {res['rotulo_modelo']}  |  R2 (fit) = {res['r2_tendencia']:.4f}",
        ]
        linhas += [f"  {d}" for d in res["descricao_modelo"]]

    if incluir_previsao and res["modelo_tendencia"] is not None:
        linhas += [
            "",
            f"[Previsao] Estimativa para os proximos {res['anos_previsao']} anos (mensal):",
        ]
        linhas += [f"  {frac_para_mes_ano(xf, ano_base)}: {formata_numero(val)}"
                   for xf, val in zip(res["x_futuros"], res["estimativa_futura"])]

    return "\n".join(linhas)

def resumo_mortalidade(res):
    """Resumo da taxa diária de mortalidade: óbitos (Holt-Winters) + Pi por ano."""
    d = res["diaria"]
    linhas = [
        resumo_texto(res["res_obitos"], incluir_previsao=False),
        "",
        f"[Populacao] '{res['arquivo_populacao']}', linha '{LINHA_POPULACAO}' "
        f"({res['populacao_anual'].index[0]}-{res['populacao_anual'].index[-1]})",
        f"  Histopolacao anual -> diaria  |  maior |media do ano - valor IBGE| = "
        f"{res['erro_populacao']:.2e}",
        "",
        f"[Mortalidade] Pi = obitos/dia / populacao/dia (por pessoa por dia)  |  "
        f"{d['dia'].iloc[0]} a {d['dia'].iloc[-1]}  |  "
        f"{int(d['observado'].sum())} dias observados, {int((~d['observado']).sum())} previstos",
    ]
    for ano, l in res["mortalidade_anual"].iterrows():
        linhas.append(f"  {ano}: Pi medio = {l['Pi_medio']:.4e}  "
                      f"({formata_numero(l['Pi_medio'] * 1e5, 3)} por 100 mil/dia)  |  "
                      f"obitos = {formata_numero(l['obitos'])} em {int(l['dias'])} dias  |  "
                      f"populacao media = {formata_numero(l['populacao_media'])}")
    return "\n".join(linhas)


# =============================================================================
# 5. PLOT (resolução diária, dados mensais no scatter)
# =============================================================================

def formata_tooltip(sel):
    """
    Tooltip interativo: clique para ver Mês/Ano e o valor por dia. O eixo y é
    por dia: nos pontos mensais é a média diária (total / dias daquele mês, 28 a
    31), e o tooltip mostra também o total do mês; nas curvas é a taxa diária.
    """
    x_real, y_val = sel.target[0], sel.target[1]
    reais = getattr(sel.artist, "valores_reais", None)  # pontos presos na borda do eixo
    if reais is not None:
        y_val = reais[sel.index]
    ano, mes = ano_mes(x_real)
    if getattr(sel.artist, "taxa", False):
        sel.annotation.set_text(
            f"{NOMES_MES_CURTO[mes-1]}/{ano}\n"
            f"Taxa/dia: {formata_numero(y_val, 1)}"
        )
        return
    dias     = calendar.monthrange(ano, mes)[1]  # ja considera ano bissexto
    sel.annotation.set_text(
        f"{NOMES_MES_CURTO[mes-1]}/{ano}\n"
        f"Media/dia ({dias} dias): {formata_numero(y_val, 1)}\n"
        f"Total do mes: {formata_numero(y_val * dias)}"
    )

def _dias_do_mes(x_absoluto):
    """Dias de calendário do mês de cada x (ano fracionário)."""
    return np.array([calendar.monthrange(*ano_mes(v))[1]
                     for v in np.atleast_1d(x_absoluto)], dtype=float)

def montar_figura(res, fig=None, titulo=None, label_y=None, figsize=(14, 6)):
    """
    Desenha a análise (scatter mensal + spline + projeção) em uma figura.

    fig=None cria a figura via pyplot (uso em script); passando uma Figure já
    existente (ex.: embutida na interface), desenha nela sem usar o pyplot.
    Retorna (fig, ax, cursor) - guarde o cursor para o tooltip seguir ativo.
    """
    if res.get("tipo") == "mortalidade":
        return montar_figura_mortalidade(res, fig=fig, titulo=titulo, figsize=figsize)
    ano_base = res["ano_base"]

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    # Interpolacao e projecao desenhadas em resolucao diaria. A histospline vai
    # da borda inicial do 1o mes a final do ultimo e vale NaN nos periodos
    # excluidos (sem total para conservar), o que interrompe a linha ali.
    # A mesma curva cobre a projecao (meses previstos pelo Holt-Winters e depois
    # desagregados), desenhada em outra cor a partir do fim do dado real.
    spline  = res["spline"]
    x_plot  = np.arange(spline.trechos[0][0], spline.fim_observado, 1/365)
    y_plot  = spline(x_plot)
    extrapola = res["modelo_tendencia"] is not None
    if extrapola:
        x_fut = np.arange(spline.fim_observado, spline.trechos[-1][1], 1/365)
        y_fut = spline(x_fut)
    else:
        x_fut = y_fut = np.array([])

    periodos = res.get("excluir_periodo")

    # Eixo y em unidades POR DIA, o mesmo da curva: os totais mensais (dados,
    # excluidos e previsao do Holt-Winters) entram como media diaria do mes
    x_fut_mes   = res["x_futuros"] + ano_base
    y_mensal    = res["y_mensal"] / _dias_do_mes(res["x_mensal"])
    y_excluidos = (res["y_excluido"] / _dias_do_mes(res["x_excluido"])
                   if len(res["x_excluido"]) else res["y_excluido"])
    y_prev_mes  = (res["estimativa_futura"] / _dias_do_mes(x_fut_mes)
                   if len(x_fut_mes) else res["estimativa_futura"])

    # Escala do eixo y pelo que entra no modelo (dados usados, spline e projeção),
    # com a mesma folga de 5% do autoscale. Um pico no período excluído - 1,97 mi
    # em ago/2019 nas vacinas, ~5x um mês típico - esticava o eixo e espremia todo
    # o resto no rodapé do gráfico.
    usados       = np.concatenate([y_mensal, y_plot[np.isfinite(y_plot)],
                                   y_prev_mes, y_fut])
    folga        = 0.05 * (usados.max() - usados.min())
    y_min, y_max = usados.min() - folga, usados.max() + folga

    if periodos:
        x_exc, y_exc = res["x_excluido"], y_excluidos
        dentro = (y_exc >= y_min) & (y_exc <= y_max)
        ax.scatter(x_exc[dentro], y_exc[dentro],
                   facecolors="none", edgecolors="gray", zorder=4, s=22,
                   label=f"Excluidos ({descrever_periodos(periodos)})")
        # Excluídos fora da escala: triângulo preso na borda; o tooltip mostra o
        # valor real (formata_tooltip lê `valores_reais`)
        for fora, borda, marcador, rotulo in (
                (y_exc > y_max, y_max, "^", "acima da escala, ate"),
                (y_exc < y_min, y_min, "v", "abaixo da escala, ate")):
            if fora.any():
                extremo = y_exc[fora].max() if marcador == "^" else y_exc[fora].min()
                pontos  = ax.scatter(x_exc[fora], np.full(fora.sum(), borda), marker=marcador,
                                     color="gray", zorder=4, s=30, clip_on=False,
                                     label=f"Excluidos {rotulo} {formata_numero(extremo)}")
                pontos.valores_reais = y_exc[fora]
        for ini, fim in periodos:
            ax.axvspan(ini, fim, color="gray", alpha=0.08, zorder=0)

    ax.scatter(res["x_mensal"], y_mensal,
               color="crimson", zorder=5, s=18, alpha=0.7, label="Dados mensais (media/dia)")
    linha, = ax.plot(x_plot + ano_base, y_plot, color="royalblue", lw=1.5,
                     label="Histospline (taxa diaria)")
    linha.taxa = True   # formata_tooltip: valor e taxa, nao media do mes
    if extrapola:
        ax.scatter(x_fut_mes, y_prev_mes,
                   color="purple", zorder=5, marker="x", s=30, alpha=0.8,
                   label="Holt-Winters mensal (media/dia)")
        proj, = ax.plot(x_fut + ano_base, y_fut, color="purple", lw=2, linestyle="--",
                        label="Previsao em taxa diaria (Holt-Winters + histospline)")
        proj.taxa = True

    # Área integrada (res["integral"]): acima de zero soma, abaixo subtrai
    areas = []
    if res.get("integral"):
        for i, (a, b) in enumerate(res["integral"]["trechos"]):
            xa = np.linspace(a - ano_base, b - ano_base, 4000)
            ya = res["spline"](xa)
            for cond, cor, rot in ((ya >= 0, "seagreen", "Área positiva"),
                                   (ya <= 0, "indianred", "Área negativa")):
                areas.append(ax.fill_between(xa + ano_base, ya, 0, where=cond,
                                             interpolate=True, color=cor, alpha=0.18,
                                             zorder=1, label=rot if i == 0 else None))
    ax.set_ylim(y_min, y_max)
    # Séries de diferença (ex.: saídas - entradas) trocam de sinal: marca o zero
    if y_min < 0 < y_max:
        ax.axhline(0, color="black", lw=0.8, alpha=0.6, zorder=1)

    # O período excluído vai numa segunda linha: somado a um nome de série longo,
    # ele passava da largura da figura e o título saía cortado
    titulo_padrao = (f"Interpolação e Extrapolação - {res['col_valor']}" if extrapola
                     else f"Interpolação - {res['col_valor']}")
    if periodos:
        titulo_padrao += f"\n(sem o período excluído: {descrever_periodos(periodos)})"
    ax.set_title(titulo or titulo_padrao, fontsize=14, fontweight="bold")
    ax.set_xlabel("Ano")
    ax.set_ylabel(textwrap.fill(label_y or f"{res['col_valor']} por dia", 40))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    # As áreas sombreadas ficam fora do tooltip (não são pontos mensais)
    alvos  = [a for a in ax.collections + ax.lines if a not in areas]
    cursor = mplcursors.cursor(alvos, hover=False)
    cursor.connect("add", formata_tooltip)

    return fig, ax, cursor

def montar_figura_mortalidade(res, fig=None, titulo=None, figsize=(14, 8)):
    """
    Três painéis com o mesmo eixo de datas: óbitos/dia, população/dia (com os
    valores anuais do IBGE no meio de cada ano) e Pi em óbitos por 100 mil
    habitantes por dia. Observado em azul, previsão em roxo tracejado; períodos
    excluídos do Holt-Winters ficam vazios (sem total para histopolar).
    Retorna (fig, eixos, cursor), como `montar_figura`.
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clear()
    eixos = fig.subplots(3, 1, sharex=True)

    # Grade diaria completa: os dias sem dado (periodo excluido) viram NaN e
    # interrompem a linha em vez de liga-la por cima do buraco
    d = res["diaria"].copy()
    d.index = pd.to_datetime(d.pop("dia"))
    d = d.asfreq("D")

    paineis = (("obitos",    1.0,  "Óbitos/dia",                  0),
               ("populacao", 1e-6, "População (milhões)",         3),
               ("valor",     1e5,  "Pi (por 100 mil hab. por dia)", 3))
    for ax, (coluna, escala, rotulo, casas) in zip(eixos, paineis):
        for previsto, cor, estilo, legenda in ((False, "royalblue", "-", "observado"),
                                               (True, "purple", "--", "previsto")):
            parte = d[coluna].where(d["observado"].eq(not previsto) & d["observado"].notna())
            linha, = ax.plot(parte.index, parte * escala, estilo, color=cor, lw=1.2, label=legenda)
            linha.rotulo_tooltip = (rotulo, escala, casas, coluna == "valor")
        ax.set_ylabel(textwrap.fill(rotulo, 22))
        ax.grid(True, linestyle="--", alpha=0.5)

    # Valores anuais do IBGE no meio do ano: a media diaria de cada ano e igual a eles
    pop  = res["populacao_anual"]
    anos = pop.index[(pop.index >= d.index.min().year) & (pop.index <= d.index.max().year)]
    pontos = eixos[1].scatter(pd.to_datetime([datetime.date(a, 7, 2) for a in anos]),
                              pop.loc[anos] * 1e-6, color="crimson", s=18, zorder=5,
                              label="IBGE anual (média do ano)")
    pontos.rotulo_tooltip = ("População IBGE do ano (milhões)", 1e-6, 3, False)

    for ini, fim in res.get("excluir_periodo") or []:
        for ax in eixos:
            ax.axvspan(pd.Timestamp(_dia_para_data(ini)), pd.Timestamp(_dia_para_data(fim)),
                       color="gray", alpha=0.08, zorder=0)
    eixos[0].legend(loc="upper left")
    eixos[1].legend(loc="upper left")
    eixos[2].set_xlabel("Data")

    titulo_padrao = "Taxa diária de mortalidade - Brasil (óbitos / população)"
    if res.get("excluir_periodo"):
        titulo_padrao += (f"\n(Holt-Winters sem o período excluído: "
                          f"{descrever_periodos(res['excluir_periodo'])})")
    eixos[0].set_title(titulo or titulo_padrao, fontsize=13, fontweight="bold")
    fig.tight_layout()

    cursor = mplcursors.cursor([a for ax in eixos for a in ax.lines + ax.collections
                                if hasattr(a, "rotulo_tooltip")], hover=False)
    cursor.connect("add", _tooltip_mortalidade)
    return fig, eixos, cursor

def _dia_para_data(x_absoluto):
    """Ano fracionário (datas reais) -> datetime.date."""
    return datetime.date.fromordinal(int(round(float(_ano_para_dia(x_absoluto)))))

def _tooltip_mortalidade(sel):
    """Data e valor do painel; no painel de Pi, também por pessoa-dia."""
    rotulo, escala, casas, eh_pi = sel.artist.rotulo_tooltip
    data  = mdates.num2date(sel.target[0]).date()
    texto = f"{data:%d/%m/%Y}\n{rotulo}: {formata_numero(sel.target[1], casas)}"
    if eh_pi:
        texto += f"\nPi = {sel.target[1] / escala:.4e} por pessoa-dia"
    sel.annotation.set_text(texto)


# =============================================================================
# 6. EXECUÇÃO COMO SCRIPT (a interface esta em interpolate_gui.py)
# =============================================================================

def salvar_serie_diaria(res, diretorio=DIRETORIO_SAIDA_DIARIA):
    """
    Grava a série diária (observado + previsão) em '<nome>_diaria.csv', no
    mesmo formato dos CSVs do projeto (separador ';'). Com período excluído o
    nome ganha '_sem_covid' (ou '_com_exclusao', para outros períodos), para
    não sobrescrever a versão com a série inteira. Devolve o caminho.
    """
    periodos = res.get("excluir_periodo") or []
    sufixo   = ("" if not periodos else
                "_sem_covid" if periodos == unir_periodos(PERIODO_COVID) else "_com_exclusao")
    os.makedirs(diretorio, exist_ok=True)
    caminho = os.path.join(diretorio, f"{res['nome']}_diaria{sufixo}.csv")
    df = res["diaria"].copy()
    df["tipo"] = np.where(df.pop("observado"), "observado", "previsto")
    df.to_csv(caminho, sep=SEPARADOR_CSV, index=False)
    return caminho

def main():
    res = analisar_csv(ARQUIVO_CSV,
                       separador=SEPARADOR_CSV,
                       anos_previsao=QUANTIDADE_ANOS_PREVISAO)
    print(resumo_texto(res))
    print(f"\nSerie diaria gravada em '{salvar_serie_diaria(res)}'")
    montar_figura(res)
    plt.show()


if __name__ == "__main__":
    main()
