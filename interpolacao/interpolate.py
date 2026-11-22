import os
import sys
import glob
import fnmatch
import calendar
import datetime
import textwrap
import warnings
import unicodedata

import numpy as np
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import mplcursors
from sklearn.metrics import r2_score
from scipy.interpolate import PPoly
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SEIHRVS import pior_caso_poisson, formata_quantil

# =============================================================================
# CONFIGURACOES - ajuste aqui para diferentes conjuntos de dados
# =============================================================================
# Sao dois metodos, um para cada metade do trabalho:
#   INTERPOLACAO  -> linear do acumulado (total do mes dividido igualmente entre os dias)
#   EXTRAPOLACAO  -> Holt-Winters (suavizacao exponencial tripla)

# Caminhos resolvidos a partir da pasta deste arquivo (e nao do diretorio de
# execucao), para o script e a interface funcionarem de qualquer lugar
RAIZ             = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DADOS  = os.path.join(RAIZ, "..", "dados", "series_mensais")                 # pasta varrida pela interface
SEPARADOR_CSV    = ";"                                               # separador do CSV
DIRETORIO_SAIDA_DIARIA = os.path.join(RAIZ, "..", "dados", "resultados", "interpolacao")  # <nome>_diaria.csv

# Taxa diaria de mortalidade (Pi do SEIHRVS) = obitos / populacao, ambos diarios
ARQUIVO_OBITOS     = os.path.join(DIRETORIO_DADOS, "obitos_mensais_brasil.csv")
ARQUIVO_NASCIMENTOS = os.path.join(DIRETORIO_DADOS, "nascimentos_mensais_brasil.csv")  # projecao da populacao
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

# CSVs da pasta de dados que NÃO aparecem como série própria na interface
# (padrão glob sobre o nome do arquivo):
#   - doses de vacina (1ª e 2ª dose): fora desta análise;
#   - emigração e imigração: entram só pelo saldo migratório (SERIES_DIFERENCA
#     em interpolate_gui.py), que continua lendo os dois arquivos.
CSVS_IGNORADOS = ("doses_*.csv", "emigracao_*.csv", "imigracao_*.csv")

# CSVs que entram SÓ com a interpolação (total do mês dividido entre os dias):
# sem Holt-Winters e sem preenchimento da lacuna do período excluído.
# Casos importados de sarampo são contagens raras e em surtos (zeros na maior
# parte dos meses), sem a sazonalidade regular que o Holt-Winters supõe.
CSVS_SEM_HOLT_WINTERS = ("casos_importados_*.csv",)

# Casos importados de sarampo (SINAN/TABNET, país provável de infecção fora do
# Brasil, classificação final = sarampo), por ano e mês de NOTIFICAÇÃO.
#   - TABNET original (latin-1, tabela ano x mês) guardado em dados/sinan;
#   - série mensal (formato dos demais CSVs) em dados/series_mensais.
ARQUIVO_IMPORTADOS        = os.path.join(DIRETORIO_DADOS,
                                         "casos_importados_sarampo_mensais_brasil.csv")


# =============================================================================
# 1. LEITURA DOS DADOS
# =============================================================================

def listar_csvs(diretorio=DIRETORIO_DADOS):
    """Lista ordenada dos CSVs do diretório de dados, sem os de `CSVS_IGNORADOS`."""
    return sorted(arq for arq in glob.glob(os.path.join(diretorio, "*.csv"))
                  if not any(fnmatch.fnmatch(os.path.basename(arq), p) for p in CSVS_IGNORADOS))

def eh_arquivo_nascimentos(arquivo):
    """True se `arquivo` é o CSV de nascimentos (exporta os nascimentos por dia)."""
    return os.path.normcase(os.path.abspath(arquivo)) == os.path.normcase(os.path.abspath(ARQUIVO_NASCIMENTOS))

def eh_arquivo_obitos(arquivo):
    """True se `arquivo` é o CSV de óbitos (vira a taxa diária de mortalidade)."""
    return os.path.normcase(os.path.abspath(arquivo)) == os.path.normcase(os.path.abspath(ARQUIVO_OBITOS))

def eh_sem_holt_winters(arquivo):
    """True se o CSV entra só com a interpolação, sem Holt-Winters (`CSVS_SEM_HOLT_WINTERS`)."""
    return any(fnmatch.fnmatch(os.path.basename(arquivo), p) for p in CSVS_SEM_HOLT_WINTERS)

def eh_arquivo_importados(arquivo):
    """True se `arquivo` é o CSV de casos importados de sarampo."""
    return os.path.normcase(os.path.abspath(arquivo)) == os.path.normcase(os.path.abspath(ARQUIVO_IMPORTADOS))

def referencias_importacao(arquivo=ARQUIVO_IMPORTADOS, separador=SEPARADOR_CSV):
    """
    Dois cenários de referência para o λ (casos importados por ano), medidos
    na série mensal do SINAN como casos do período / anos do período (meses/12):
      "Com surto" -> todos os meses (inclui o surto de 2018-2020);
      "Sem surto" -> sem os meses de 2018 a 2020.
    Devolve {rótulo: λ}, com "Com surto" primeiro.
    """
    col_periodo, col_valor = ler_colunas(arquivo, separador)
    x, y = carregar_dados_mensais(arquivo, separador, col_periodo, col_valor)
    ano  = np.floor(x).astype(int)
    periodo = f"{frac_para_mes_ano(x[0], 0)}-{frac_para_mes_ano(x[-1], 0)}"

    def taxa(mascara):
        return float(y[mascara].sum() / (mascara.sum() / 12.0))

    return {
        f"Com surto ({periodo})":            taxa(np.ones_like(y, dtype=bool)),
        f"Sem surto 2018-2020 ({periodo})":  taxa((ano < 2018) | (ano > 2020)),
    }

def pior_caso_importados(res, lambda_ano, anos, quantil):
    """
    Projeção dos casos importados depois do último dia observado pelo PIOR CASO
    do processo de Poisson, o mesmo do SEIHRVS (`pior_caso_poisson`): total no
    horizonte = quantil q de Poisson(λ·H/365), distribuído a taxa constante
    (horizonte H = anos x 365 dias, passo de 1 dia). O observado continua
    sendo a interpolação dos totais mensais do SINAN.

    Devolve uma cópia de `res` com:
      diaria               -> observado + ε constante do pior caso nos dias
                              projetados (observado=False), o mesmo do SEIHRVS
      pior_caso_importados -> {lambda_ano, quantil, dias (datas), media, total,
                               eps_diario, acumulado_media, acumulado_pior_caso}
    """
    raiz_projeto = os.path.abspath(os.path.join(RAIZ, ".."))
    if raiz_projeto not in sys.path:
        sys.path.insert(0, raiz_projeto)

    n_dias = int(round(anos * 365))   # mesmo horizonte do SEIHRVS (365 x anos)
    pc     = pior_caso_poisson(float(lambda_ano), 0, n_dias, 1.0, float(quantil))
    d      = res["diaria"]
    inicio = d["dia"].iloc[-1] + datetime.timedelta(days=1)
    dias   = [inicio + datetime.timedelta(days=i) for i in range(n_dias)]
    fracao = np.arange(1, n_dias + 1) / n_dias
    futuro = pd.DataFrame({"dia": dias, "valor": pc["eps"], "observado": False})
    return {**res,
            "diaria": pd.concat([d, futuro], ignore_index=True),
            "pior_caso_importados": {
                "lambda_ano":          pc["lambda_ano"],
                "quantil":             pc["quantil"],
                "dias":                dias,
                "media":               pc["media"],
                "total":               pc["total"],
                "eps_diario":          pc["eps"][0] if n_dias else 0.0,
                "acumulado_media":     pc["media"] * fracao,
                "acumulado_pior_caso": pc["total"] * fracao,
            }}

def descrever_pior_caso_importados(pc):
    return (f"Pior caso {formata_quantil(pc['quantil'])} da Poisson (λ = "
            f"{formata_numero(pc['lambda_ano'], 2)} casos/ano, {pc['dias'][0]:%d/%m/%Y} a "
            f"{pc['dias'][-1]:%d/%m/%Y}): {formata_numero(pc['total'], 0)} casos importados no "
            f"período (média {formata_numero(pc['media'], 1)})  |  taxa constante de "
            f"{formata_numero(pc['eps_diario'], 4)} casos/dia no CSV diário")

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
# 2. INTERPOLACAO LINEAR DO ACUMULADO (CONSERVA OS TOTAIS MENSAIS)
# =============================================================================
# Cada ponto e o TOTAL de um mes, e nao o valor da serie num instante.
# Interpolacao em linha reta (recomendacao do orientador): o acumulado A(t) e
# ligado por retas entre as bordas dos meses, ou seja, o total de cada mes e
# dividido IGUALMENTE entre os dias daquele mes (valor diario = total / dias do
# mes). Restricoes atendidas por construcao:
#   1. integral de cada mes = total do mes (A(fim) - A(inicio));
#   2. valores nos nos iguais: A(borda) = acumulado exato dos meses;
#   3. nao negatividade: valor diario = total / dias >= 0.
# Consequencia (hipotese declarada): o valor diario e constante dentro do mes e
# muda em degrau na virada do mes; a forma dentro do mes nao vem dos dados.
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

def interpolar_acumulado(bordas_dias, totais):
    """
    Interpolação linear do acumulado: `totais[i]` é o total do intervalo
    [bordas_dias[i], bordas_dias[i+1]) (dias ordinais, intervalos encostados),
    dividido igualmente entre os dias do intervalo.

    Devolve o acumulado A(dia) (PPoly de grau 1 que passa por (bordas,
    acumulado), A(bordas[0]) = 0); a taxa é A.derivative() (total / dias, constante
    no intervalo) e o total de um dia d é A(d+1) - A(d). Serve para meses ou anos.
    """
    b = np.asarray(bordas_dias, dtype=float)
    y = np.asarray(totais, dtype=float)
    if np.any(y < 0):
        raise ValueError("A interpolação exige totais não negativos (valor diário >= 0).")
    acumulado = np.concatenate([[0.0], np.cumsum(y)])
    return PPoly(np.vstack([y / np.diff(b), acumulado[:-1]]), b)

class InterpolacaoMensal:
    """
    Taxa diária com o total de cada mês dividido igualmente entre os dias do
    mês (`interpolar_acumulado`). x em anos fracionários normalizados
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

        # Por dentro tudo em dias ordinais: o acumulado A(dia) e linear por partes
        # entre as bordas dos meses e a taxa diaria A'(dia) = total do mes / dias.
        # As bordas em x (anos) ficam para o grafico.
        self.trechos = []   # (x_inicio, x_fim, acumulado A(dia) em dias)
        self.meses   = []   # (x_inicio, x_fim, total do mes)
        self.dia_fim_observado = int(round(float(_ano_para_dia(self.fim_observado + ano_base))))
        cortes = np.flatnonzero(np.diff(idx) != 1) + 1
        for i_seg, y_seg in zip(np.split(idx, cortes), np.split(y, cortes)):
            bordas = np.array([_inicio_do_mes(k) - ano_base
                               for k in range(i_seg[0], i_seg[-1] + 2)])
            dias = np.array([datetime.date(k // 12, k % 12 + 1, 1).toordinal()
                             for k in range(i_seg[0], i_seg[-1] + 2)], dtype=float)
            acumulado = interpolar_acumulado(dias, y_seg)
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
    ficarem fora do R² e do RMSE): reta de mínimos quadrados sobre TODOS os
    meses observados, vezes (ou mais) o fator sazonal mediano daquele mês em
    relação à reta.

    Escolha medida nos dados (lacunas artificiais fora da COVID, valor real
    conhecido; 11 lacunas de 12, 24 e 36 meses por série): erro médio (MAPE)
    de 3,5% nos nascimentos e 1,9% nos óbitos, contra 3,7% e 3,9% do método
    anterior (nível interpolado entre as bordas x sazonal). Só 1 dessas
    lacunas tem os 36 meses da COVID. Hipótese: a tendência do período é
    aproximadamente linear - os dados não mostram se vale dentro da lacuna.
    """
    idx   = np.rint(np.asarray(x, dtype=float) * 12 - 0.5).astype(int)  # mes 0 = jan do ano-base
    ordem = np.argsort(idx)
    idx   = idx[ordem]

    grade     = np.arange(idx[0], idx[-1] + 1)
    serie     = np.interp(grade, idx, np.asarray(y, dtype=float)[ordem])
    observado = np.isin(grade, idx)

    faltantes = ~observado
    if faltantes.any():
        # Reta ajustada so nos meses reais, avaliada na grade inteira
        reta   = np.polyval(np.polyfit(grade[observado], serie[observado], 1), grade)
        desvio = serie / reta if multiplicativa else serie - reta
        meses  = grade % periodo
        tipico = (pd.Series(desvio[observado]).groupby(meses[observado]).median()
                  .reindex(range(periodo)).fillna(1.0 if multiplicativa else 0.0)
                  .to_numpy())[meses[faltantes]]
        serie[faltantes] = (reta[faltantes] * tipico if multiplicativa
                            else reta[faltantes] + tipico)

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
                 excluir_periodo=EXCLUIR_PERIODO_PADRAO,
                 extrapolar=True):
    """
    Interpolação linear do acumulado (total de cada mês dividido igualmente
    entre os dias) e, se `extrapolar`, extrapolação por Holt-Winters.
    Retorna um dicionário com os dados, os modelos e as métricas do arquivo.
    excluir_periodo: None, um (ini, fim) ou uma lista deles.
    CSVs de `CSVS_SEM_HOLT_WINTERS` ignoram `extrapolar` (só a interpolação).
    """
    somente = eh_sem_holt_winters(arquivo)
    col_periodo, col_valor = ler_colunas(arquivo, separador)
    x_mensal, y_mensal = carregar_dados_mensais(arquivo, separador, col_periodo, col_valor)
    return analisar_serie(x_mensal, y_mensal, arquivo, col_periodo, col_valor,
                          anos_previsao, excluir_periodo,
                          extrapolar=extrapolar and not somente,
                          preencher_lacuna=not somente)

def analisar_diferenca(arquivo_a, arquivo_b,
                       separador=SEPARADOR_CSV,
                       excluir_periodo=EXCLUIR_PERIODO_PADRAO,
                       nome=None):
    """
    Saldo A - B (ex.: emigração - imigração) a partir das séries INTERPOLADAS.

    Cada CSV é interpolado separadamente (sem extrapolação): sai a
    taxa diária de A e a de B, cada uma conservando os totais mensais do seu
    CSV. O saldo diário é a diferença das duas, só nos dias cobertos por ambas,
    e res["integral"] é a integral desse saldo (∫ (a - b) dt, t em dias).

    Como cada valor diário é A(d+1) - A(d) do acumulado, a soma dos dias é a
    integral exata da curva; e como as duas interpolações conservam o total de
    cada mês, essa integral é igual à soma dos saldos mensais (vai junto como
    conferência).
    """
    res_a = analisar_csv(arquivo_a, separador=separador,
                         excluir_periodo=excluir_periodo, extrapolar=False)
    res_b = analisar_csv(arquivo_b, separador=separador,
                         excluir_periodo=excluir_periodo, extrapolar=False)

    diaria = res_a["diaria"][["dia", "valor"]].merge(
        res_b["diaria"][["dia", "valor"]], on="dia", suffixes=("_a", "_b"))
    if diaria.empty:
        raise ValueError(f"'{arquivo_a}' e '{arquivo_b}' nao tem nenhum dia em comum.")
    diaria = pd.DataFrame({
        "dia":       diaria["dia"],
        "valor":     diaria["valor_a"] - diaria["valor_b"],   # saldo por dia
        res_a["nome"]: diaria["valor_a"],
        res_b["nome"]: diaria["valor_b"],
        "observado": True,
    }).sort_values("dia", ignore_index=True)

    # Saldo mensal (totais A - B), casando os meses pelo indice inteiro
    def _saldo_mensal(xa, ya, xb, yb):
        ia_m = np.rint(np.asarray(xa) * 12 - 0.5).astype(int)
        ib_m = np.rint(np.asarray(xb) * 12 - 0.5).astype(int)
        _, ia, ib = np.intersect1d(ia_m, ib_m, return_indices=True)
        return np.asarray(xa)[ia], np.asarray(ya)[ia] - np.asarray(yb)[ib]

    x_mensal, y_mensal = _saldo_mensal(res_a["x_mensal"], res_a["y_mensal"],
                                       res_b["x_mensal"], res_b["y_mensal"])
    x_exc, y_exc = (_saldo_mensal(res_a["x_excluido"], res_a["y_excluido"],
                                  res_b["x_excluido"], res_b["y_excluido"])
                    if len(res_a["x_excluido"]) and len(res_b["x_excluido"])
                    else (np.array([]), np.array([])))

    # Trechos de dias consecutivos (o periodo excluido abre um buraco)
    ordinais = np.array([d.toordinal() for d in diaria["dia"]])
    cortes   = np.flatnonzero(np.diff(ordinais) != 1) + 1
    trechos  = [(float(_dia_para_ano(seg[0])), float(_dia_para_ano(seg[-1] + 1)))
                for seg in np.split(ordinais, cortes)]

    nome_a = res_a["col_valor"]
    nome_b = res_b["col_valor"]
    ordem  = np.argsort(x_mensal)
    ano_base = int(np.floor(x_mensal.min()))
    return {
        "tipo":              "saldo",
        "arquivo":           nome or f"{os.path.basename(arquivo_a)} - {os.path.basename(arquivo_b)}",
        "nome":              nome or f"{res_a['nome']}-{res_b['nome']}",
        "col_periodo":       "Ano/mes",
        "col_valor":         f"{nome_a} - {nome_b}",
        "nome_a":            nome_a,           # rotulo (coluna de valor do CSV)
        "nome_b":            nome_b,
        "coluna_a":          res_a["nome"],    # coluna da serie diaria
        "coluna_b":          res_b["nome"],
        "res_a":             res_a,
        "res_b":             res_b,
        "x_mensal":          x_mensal[ordem],
        "y_mensal":          y_mensal[ordem],
        "x_excluido":        x_exc,
        "y_excluido":        y_exc,
        "excluir_periodo":   res_a["excluir_periodo"],
        "ano_base":          ano_base,
        "x":                 x_mensal[ordem] - ano_base,
        "y":                 y_mensal[ordem],
        "diaria":            diaria,
        "erro_conservacao":  max(res_a["erro_conservacao"], res_b["erro_conservacao"]),
        "rotulo_modelo":     None,
        "descricao_modelo":  [],
        "r2_tendencia":      None,
        "modelo_tendencia":  None,
        "anos_previsao":     0,
        "x_futuros":         np.array([]),
        "estimativa_futura": np.array([]),
        "integral": {
            "integral":       float(diaria["valor"].sum()),   # ∫ saldo dt, t em dias
            "trechos":        trechos,
            "meses_cobertos": int(len(x_mensal)),
        },
    }

def analisar_serie(x_mensal, y_mensal, arquivo, col_periodo, col_valor,
                   anos_previsao=QUANTIDADE_ANOS_PREVISAO,
                   excluir_periodo=EXCLUIR_PERIODO_PADRAO,
                   extrapolar=True, preencher_lacuna=True):
    """Núcleo de analisar_csv: recebe a série mensal (x em ano fracionário, y)."""
    # Descarta os períodos excluídos (ex.: choque da COVID-19, picos de surto). Os
    # pontos saem da interpolação e do ajuste, mas continuam guardados para o gráfico.
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

    # --- Interpolacao linear do acumulado, conserva o total de cada mes ------
    interpolacao = InterpolacaoMensal(x, y, ano_base)

    # Preenchimento da lacuna do periodo excluido (reta global x sazonal, o mesmo
    # que o Holt-Winters usa em _completar_serie_mensal), so para exibir: os meses
    # reconstruidos nao entram na interpolacao nem nas metricas.
    x_rec = y_rec = np.array([])
    if periodos and preencher_lacuna:
        mult = SAZONALIDADE_HW == "multiplicativa" and float(np.min(y)) > 0
        grade, completa, obs = _completar_serie_mensal(x, y, PERIODO_SAZONAL, mult)
        x_rec = (grade[~obs] + 0.5) / 12 + ano_base   # meio do mes, ano absoluto
        y_rec = completa[~obs]

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
        "interpolacao":      interpolacao,
        "erro_conservacao":  interpolacao.erro_conservacao(),   # ~0: confere a construcao
        "diaria":            interpolacao.diario(),             # dia, valor, observado
        "x_reconstruido":    x_rec,   # meses preenchidos na lacuna (ano absoluto)
        "y_reconstruido":    y_rec,   # totais mensais preenchidos
    }

    # Sem extrapolacao: so a interpolacao (modelo_tendencia = None sinaliza ao
    # grafico, ao resumo e a interface que nao ha curva projetada)
    if not extrapolar:
        return {**base,
                "rotulo_modelo":     None,
                "descricao_modelo":  [],
                "r2_tendencia":      None,
                "rmse_tendencia":    None,
                "modelo_tendencia":  None,
                "anos_previsao":     0,
                "x_futuros":         np.array([]),
                "estimativa_futura": np.array([])}

    # --- Extrapolacao: Holt-Winters, em passos mensais (1/12 de ano) ---------
    # contagem inteira de meses: np.arange com passo 1/12 as vezes inclui um a mais
    x_futuros = x[-1] + np.arange(1, int(round(anos_previsao * 12)) + 1) / 12
    modelo_tendencia, hw = ajustar_holt_winters(
        x, y, passos_futuros=int(round(anos_previsao * PERIODO_SAZONAL)))

    # R2 e RMSE medidos SO nos meses realmente observados (os reconstruidos para
    # fechar o buraco do periodo excluido nao entram na conta), sobre os totais
    # mensais: o RMSE sai na unidade do CSV por mes.
    ajustado  = modelo_tendencia(x)
    r2_tend   = r2_score(y, ajustado)
    rmse_tend = float(np.sqrt(np.mean((y - ajustado) ** 2)))

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
            f"{hw['meses_reconstruidos']} reconstruidos (reta global x "
            f"sazonalidade tipica) so para manter o ciclo alinhado - eles nao "
            f"entram no R2 nem no RMSE"
        )

    # Ordem: Holt-Winters sobre os totais mensais REAIS (acima) e so depois a
    # desagregacao por dia, do passado e da projecao juntos. O contrario (rodar
    # o Holt-Winters sobre dias inventados) poria no modelo a forma intra-mes
    # suposta pela interpolacao, que nao vem dos dados.
    estimativa   = modelo_tendencia(x_futuros)
    interpolacao = InterpolacaoMensal(np.concatenate([x, x_futuros]),
                                      np.concatenate([y, estimativa]),
                                      ano_base, x_ultimo_observado=x[-1])

    return {
        **base,
        "interpolacao":      interpolacao,
        "erro_conservacao":  interpolacao.erro_conservacao(),
        "diaria":            interpolacao.diario(),   # observado + previsao, por dia
        "rotulo_modelo":     rotulo,
        "descricao_modelo":  descricao,
        "r2_tendencia":      r2_tend,
        "rmse_tendencia":    rmse_tend,
        "modelo_tendencia":  modelo_tendencia,
        "anos_previsao":     anos_previsao,
        "x_futuros":         x_futuros,
        "estimativa_futura": estimativa,
    }

# --- Taxa diaria de mortalidade ---------------------------------------------
# Pi(dia) = obitos(dia) / populacao(dia), em obitos por pessoa por dia (unidade
# do Pi do SEIHRVS), sobre a populacao brasileira inteira.
#   Obitos e nascimentos: total mensal dividido entre os dias (analisar_csv)
#     e, se a extrapolacao estiver ligada, Holt-Winters nos totais mensais de
#     cada serie, desagregado por dia do mesmo jeito.
#   Populacao no periodo OBSERVADO: populacao anual total do IBGE, constante
#     em todos os dias do ano (populacao_diaria).
#   Populacao no periodo PROJETADO: parte do valor diario do IBGE no ultimo dia
#     observado e avanca pelo balanco P(d+1) = P(d) + nascimentos(d) - obitos(d),
#     com nascimentos e obitos diarios do Holt-Winters.
# Hipoteses (os dados nao permitem medi-las):
#   - o valor anual do IBGE e a MEDIA da populacao no ano: a soma dos dias do ano
#     da valor x dias do ano. O arquivo nao informa a data de referencia;
#   - obitos de todas as causas sobre a populacao residente total: Pi igual para
#     todas as idades e compartimentos, como no modelo;
#   - a projecao da populacao e FECHADA: so nascimentos - obitos, sem o saldo
#     migratorio;
#   - obitos e nascimentos sao constantes dentro do mes, e a populacao dentro do
#     ano (interpolacao linear do acumulado): a forma intra-mes/ano nao vem dos
#     dados.

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
    Anual -> diária pela interpolação linear do acumulado: cada ano entra com
    população x dias do ano (pessoa-dia), dividida igualmente entre os dias,
    então o valor de um dia é a população do ano. Devolve Series indexada pela data.
    """
    anos = pop_anual.index.to_numpy()
    if not np.all(np.diff(anos) == 1):
        raise ValueError("A projeção anual precisa de anos consecutivos.")
    bordas = np.array([datetime.date(int(a), 1, 1).toordinal() for a in np.append(anos, anos[-1] + 1)],
                      dtype=float)
    acumulado = interpolar_acumulado(bordas, pop_anual.to_numpy() * np.diff(bordas))
    dias = np.arange(bordas[0], bordas[-1])
    return pd.Series(acumulado(dias + 1) - acumulado(dias),
                     index=pd.to_datetime([datetime.date.fromordinal(int(d)) for d in dias]),
                     name="populacao")

def analisar_mortalidade(anos_previsao=QUANTIDADE_ANOS_PREVISAO,
                         excluir_periodo=EXCLUIR_PERIODO_PADRAO,
                         arquivo_obitos=ARQUIVO_OBITOS,
                         arquivo_populacao=ARQUIVO_POPULACAO,
                         extrapolar=True,
                         arquivo_nascimentos=ARQUIVO_NASCIMENTOS):
    """
    Taxa diária de mortalidade Pi = óbitos / população (por pessoa por dia).
    Devolve o resultado da análise dos óbitos (interpolação e, se `extrapolar`,
    Holt-Winters) com `diaria` trocada por dia, valor (= Pi), obitos,
    nascimentos, populacao, populacao_ibge e observado, e:
      mortalidade_anual -> por ano: óbitos, nascimentos, população média, Pi médio
      erro_populacao    -> maior |média diária do ano - valor do IBGE| (~0)
      res_obitos, res_nascimentos -> as análises originais
      dia_ancora        -> último dia observado (início da projeção da população)
    """
    kw = dict(anos_previsao=anos_previsao, excluir_periodo=excluir_periodo,
              extrapolar=extrapolar)
    res_obitos = analisar_csv(arquivo_obitos, **kw)
    res_nasc   = analisar_csv(arquivo_nascimentos, **kw)
    pop_anual  = ler_populacao_anual(arquivo_populacao)
    pop        = populacao_diaria(pop_anual)

    # So os dias cobertos pelas duas series; observado = observado nas duas
    juntos = res_obitos["diaria"].merge(res_nasc["diaria"], on="dia",
                                        suffixes=("_obitos", "_nasc"))
    if juntos.empty:
        raise ValueError("Óbitos e nascimentos não têm nenhum dia em comum.")
    datas     = pd.to_datetime(juntos["dia"])
    obitos    = juntos["valor_obitos"].to_numpy()
    nasc      = juntos["valor_nasc"].to_numpy()
    observado = (juntos["observado_obitos"] & juntos["observado_nasc"]).to_numpy()

    faltando = pd.DatetimeIndex(datas[observado]).difference(pop.index)
    if len(faltando):
        raise ValueError(f"A população do IBGE não cobre {len(faltando)} dias "
                         f"({faltando.min().date()} a {faltando.max().date()}).")
    pop_ibge  = pop.reindex(datas).to_numpy()   # NaN se o IBGE nao cobrir o dia
    populacao = pop_ibge.copy()

    # Projecao: a partir do ultimo dia observado, P(d+1) = P(d) + B(d) - D(d)
    i_ancora  = int(np.flatnonzero(observado)[-1])
    projetado = np.arange(len(observado)) > i_ancora
    if projetado.any():
        if not np.all(np.diff(juntos["dia"].map(datetime.date.toordinal).to_numpy()[i_ancora:]) == 1):
            raise ValueError("A projeção precisa de dias consecutivos após o último dia observado.")
        saldo = (nasc - obitos)[i_ancora:-1]          # balanco do dia anterior
        populacao[projetado] = pop_ibge[i_ancora] + np.cumsum(saldo)

    diaria = pd.DataFrame({
        "dia":            juntos["dia"],
        "valor":          obitos / populacao,   # Pi, obitos por pessoa por dia
        "obitos":         obitos,
        "nascimentos":    nasc,
        "populacao":      populacao,
        "populacao_ibge": pop_ibge,
        "observado":      observado,
    })

    ano = datas.dt.year.to_numpy()
    g   = diaria.groupby(ano)
    anual = pd.DataFrame({
        "dias":            g.size(),
        "obitos":          g["obitos"].sum(),
        "nascimentos":     g["nascimentos"].sum(),
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
        "rotulo_modelo":     (f"{res_obitos['rotulo_modelo']} dos óbitos"
                              if res_obitos["rotulo_modelo"] else None),
        "diaria":            diaria,
        "mortalidade_anual": anual,
        "populacao_anual":   pop_anual,
        "erro_populacao":    float((media_ano - pop_anual).abs().max()),
        "arquivo_obitos":    arquivo_obitos,
        "arquivo_populacao": arquivo_populacao,
        "arquivo_nascimentos": arquivo_nascimentos,
        "erro_conservacao":  max(res_obitos["erro_conservacao"], res_nasc["erro_conservacao"]),
        "res_obitos":        res_obitos,
        "res_nascimentos":   res_nasc,
        "dia_ancora":        juntos["dia"].iloc[i_ancora],
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
    if res.get("tipo") == "saldo":
        return resumo_saldo(res)
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
        f"[Interpolacao] Linear do acumulado (total do mes / dias)  |  maior erro do total "
        f"mensal = {res['erro_conservacao']:.2e}",
    ]
    if res["modelo_tendencia"] is not None:
        linhas += [
            "",
            f"[Extrapolacao] {res['rotulo_modelo']}  |  R2 (fit) = {res['r2_tendencia']:.4f}  |  "
            f"RMSE = {formata_numero(res['rmse_tendencia'])} /mes",
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

def resumo_saldo(res):
    """Resumo do saldo: as duas interpolações e a integral do saldo diário."""
    integ = res["integral"]
    d     = res["diaria"]
    return "\n".join([
        resumo_texto(res["res_a"], incluir_previsao=False),
        "",
        resumo_texto(res["res_b"], incluir_previsao=False),
        "",
        f"[Saldo] {res['col_valor']} por dia = diferenca das duas interpolacoes  |  "
        f"{d['dia'].iloc[0]} a {d['dia'].iloc[-1]} ({len(d)} dias)",
        f"[Integral do saldo] {descrever_integral(integ)}"
    ])

def resumo_mortalidade(res):
    """Resumo da taxa diária de mortalidade: óbitos, nascimentos, população e Pi por ano."""
    d = res["diaria"]
    linhas = [
        resumo_texto(res["res_obitos"], incluir_previsao=False),
        "",
        resumo_texto(res["res_nascimentos"], incluir_previsao=False),
        "",
        f"[Populacao] '{res['arquivo_populacao']}', linha '{LINHA_POPULACAO}' "
        f"({res['populacao_anual'].index[0]}-{res['populacao_anual'].index[-1]})",
        f"  Anual -> diaria (constante no ano) ate {res['dia_ancora']}  |  maior |media do ano - "
        f"valor IBGE| = {res['erro_populacao']:.2e}",
    ]
    if (~d["observado"]).any():
        linhas.append(f"  Depois de {res['dia_ancora']}: P(d+1) = P(d) + nascimentos(d) - obitos(d) "
                      f"(Holt-Winters; sem saldo migratorio)")
    linhas += [
        "",
        f"[Mortalidade] Pi = obitos/dia / populacao/dia (obitos por pessoa por dia)  |  "
        f"{d['dia'].iloc[0]} a {d['dia'].iloc[-1]}  |  "
        f"{int(d['observado'].sum())} dias observados, {int((~d['observado']).sum())} previstos",
    ]
    for ano, l in res["mortalidade_anual"].iterrows():
        linhas.append(f"  {ano}: Pi medio = {l['Pi_medio']:.4e} por pessoa por dia  |  "
                      f"obitos = {formata_numero(l['obitos'])}  |  "
                      f"nascimentos = {formata_numero(l['nascimentos'])}  |  "
                      f"em {int(l['dias'])} dias  |  "
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
    Desenha a análise (scatter mensal + interpolação + projeção) em uma figura.

    fig=None cria a figura via pyplot (uso em script); passando uma Figure já
    existente (ex.: embutida na interface), desenha nela sem usar o pyplot.
    Retorna (fig, ax, cursor) - guarde o cursor para o tooltip seguir ativo.
    """
    if res.get("tipo") == "mortalidade":
        return montar_figura_mortalidade(res, fig=fig, titulo=titulo, figsize=figsize)
    if res.get("tipo") == "saldo":
        return montar_figura_saldo(res, fig=fig, titulo=titulo, figsize=figsize)
    if res.get("pior_caso_importados"):
        return montar_figura_importados_pior_caso(res, fig=fig, titulo=titulo, figsize=figsize)
    ano_base = res["ano_base"]

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    # Interpolacao e projecao desenhadas em resolucao diaria. A interpolacao vai
    # da borda inicial do 1o mes a final do ultimo e vale NaN nos periodos
    # excluidos (sem total para conservar), o que interrompe a linha ali.
    # A mesma curva cobre a projecao (meses previstos pelo Holt-Winters e depois
    # desagregados), desenhada em outra cor a partir do fim do dado real.
    interp  = res["interpolacao"]
    x_plot  = np.arange(interp.trechos[0][0], interp.fim_observado, 1/365)
    y_plot  = interp(x_plot)
    extrapola = res["modelo_tendencia"] is not None
    if extrapola:
        x_fut = np.arange(interp.fim_observado, interp.trechos[-1][1], 1/365)
        y_fut = interp(x_fut)
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
    x_rec       = res.get("x_reconstruido", np.array([]))
    y_rec       = (res["y_reconstruido"] / _dias_do_mes(x_rec)
                   if len(x_rec) else np.array([]))

    # Escala do eixo y pelo que entra no modelo (dados usados, interpolacao, lacuna
    # preenchida e projeção), com a mesma folga de 5% do autoscale: um pico no
    # período excluído esticaria o eixo e espremeria todo o resto no rodapé.
    usados       = np.concatenate([y_mensal, y_plot[np.isfinite(y_plot)],
                                   y_prev_mes, y_fut, y_rec])
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
    if len(x_rec):
        ax.plot(x_rec, y_rec, color="darkorange", lw=1.2, linestyle=":", marker="D",
                markersize=4, zorder=5,
                label="Lacuna preenchida: reta global × sazonal (média/dia)")
    linha, = ax.plot(x_plot + ano_base, y_plot, color="royalblue", lw=1.5,
                     label="Interpolacao linear do acumulado (taxa diaria)")
    linha.taxa = True   # formata_tooltip: valor e taxa, nao media do mes
    if extrapola:
        ax.scatter(x_fut_mes, y_prev_mes,
                   color="purple", zorder=5, marker="x", s=30, alpha=0.8,
                   label="Holt-Winters mensal (media/dia)")
        proj, = ax.plot(x_fut + ano_base, y_fut, color="purple", lw=2, linestyle="--",
                        label="Previsao em taxa diaria (Holt-Winters / dias do mes)")
        proj.taxa = True

    ax.set_ylim(y_min, y_max)
    if y_min < 0 < y_max:
        ax.axhline(0, color="black", lw=0.8, alpha=0.6, zorder=1)

    # O período excluído vai numa segunda linha: somado a um nome de série longo,
    # ele passava da largura da figura e o título saía cortado
    titulo_padrao = (f"Interpolação e Extrapolação - {res['col_valor']}" if extrapola
                     else f"Interpolação - {res['col_valor']}")
    if periodos:
        titulo_padrao += f"\n(sem o período da COVID-19: {descrever_periodos(periodos)})"
    ax.set_title(titulo or titulo_padrao, fontsize=14, fontweight="bold")
    ax.set_xlabel("Ano")
    ax.set_ylabel(textwrap.fill(label_y or f"{res['col_valor']} por dia", 40))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    cursor = mplcursors.cursor(ax.collections + ax.lines, hover=False)
    cursor.connect("add", formata_tooltip)

    return fig, ax, cursor

def montar_figura_mortalidade(res, fig=None, titulo=None, figsize=(14, 8)):
    """
    Três painéis com o mesmo eixo de datas: nascimentos e óbitos por dia,
    população brasileira total por dia (IBGE constante no ano no observado; na
    previsão, população + nascimentos - óbitos) e Pi em óbitos por pessoa por
    dia. Observado em linha cheia, previsão tracejada; períodos excluídos ficam
    vazios (sem total para interpolar).
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
    obs  = d["observado"].eq(True)
    prev = d["observado"].eq(False)

    def _linha(ax, serie, mascara, cor, estilo, legenda, rotulo, casas, eh_pi=False):
        linha, = ax.plot(serie.index, serie.where(mascara), estilo, color=cor, lw=1.2, label=legenda)
        linha.rotulo_tooltip = (rotulo, 1.0, casas, eh_pi)

    # --- Painel 1: nascimentos e óbitos por dia ------------------------------
    for coluna, cor, rotulo in (("nascimentos", "seagreen", "Nascimentos/dia"),
                                ("obitos", "royalblue", "Óbitos/dia")):
        _linha(eixos[0], d[coluna], obs, cor, "-", f"{rotulo} (observado)", rotulo, 0)
        if prev.any():
            _linha(eixos[0], d[coluna], prev, cor, "--", f"{rotulo} (Holt-Winters)", rotulo, 0)
    # Lacuna do periodo excluido preenchida (reta global x sazonal), media/dia do mes
    for chave, rotulo, cor in (("res_nascimentos", "Nascimentos", "seagreen"),
                               ("res_obitos", "Óbitos", "royalblue")):
        r = res.get(chave) or {}
        x_rec = r.get("x_reconstruido", np.array([]))
        if len(x_rec):
            datas = pd.to_datetime([datetime.date(*ano_mes(v), 15) for v in x_rec])
            pts, = eixos[0].plot(datas, r["y_reconstruido"] / _dias_do_mes(x_rec), ":",
                                 color=cor, marker="D", markersize=4, markerfacecolor="darkorange",
                                 markeredgecolor="darkorange", lw=1.0,
                                 label=f"{rotulo}: lacuna preenchida (reta global × sazonal)")
            pts.rotulo_tooltip = (f"{rotulo} preenchido - média/dia do mês", 1.0, 0, False)
    eixos[0].set_ylabel("Pessoas por dia")

    # --- Painel 2: população total por dia ----------------------------------
    _linha(eixos[1], d["populacao"], obs, "royalblue", "-",
           "IBGE (constante no ano)", "População", 0)
    if prev.any():
        _linha(eixos[1], d["populacao"], prev, "purple", "--",
               "Projeção: população + nascimentos - óbitos", "População projetada", 0)
        _linha(eixos[1], d["populacao_ibge"], prev, "gray", ":",
               "Projeção do IBGE (comparação)", "População IBGE", 0)
    pop  = res["populacao_anual"]
    anos = pop.index[(pop.index >= d.index.min().year) & (pop.index <= d.index.max().year)]
    pontos = eixos[1].scatter(pd.to_datetime([datetime.date(a, 7, 2) for a in anos]),
                              pop.loc[anos], color="crimson", s=18, zorder=5,
                              label="IBGE anual (média do ano)")
    pontos.rotulo_tooltip = ("População IBGE do ano", 1.0, 0, False)
    eixos[1].set_ylabel("População (pessoas)")
    eixos[1].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: formata_numero(v)))

    # --- Painel 3: Pi por pessoa por dia ------------------------------------
    _linha(eixos[2], d["valor"], obs, "royalblue", "-", "observado",
           "Pi (óbitos por pessoa por dia)", 0, eh_pi=True)
    if prev.any():
        _linha(eixos[2], d["valor"], prev, "purple", "--", "previsto",
               "Pi (óbitos por pessoa por dia)", 0, eh_pi=True)
    eixos[2].set_ylabel(textwrap.fill("Pi (óbitos por pessoa por dia)", 22))

    for ax in eixos:
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)
    # Painel 1: legenda em duas colunas no canto direito, onde as curvas ficam
    # mais baixas (nascimentos em queda), para nao cobrir a serie de 2015-2019
    eixos[0].legend(loc="upper right", fontsize=7, ncol=2)
    eixos[0].set_ylim(top=eixos[0].get_ylim()[1] * 1.15)
    for ini, fim in res.get("excluir_periodo") or []:
        for ax in eixos:
            ax.axvspan(pd.Timestamp(_dia_para_data(ini)), pd.Timestamp(_dia_para_data(fim)),
                       color="gray", alpha=0.08, zorder=0)
    eixos[2].set_xlabel("Data")

    titulo_padrao = "Taxa diária de mortalidade - Brasil (óbitos / população)"
    if res.get("excluir_periodo"):
        titulo_padrao += (f"\n(sem o período da Covid-19: "
                          f"{descrever_periodos(res['excluir_periodo'])})")
    eixos[0].set_title(titulo or titulo_padrao, fontsize=13, fontweight="bold")
    fig.tight_layout()

    cursor = mplcursors.cursor([a for ax in eixos for a in ax.lines + ax.collections
                                if hasattr(a, "rotulo_tooltip")], hover=False)
    cursor.connect("add", _tooltip_mortalidade)
    return fig, eixos, cursor

def montar_figura_importados_pior_caso(res, fig=None, titulo=None, figsize=(14, 8)):
    """
    Casos importados com o pior caso da Poisson, em dois painéis:
      (1) observado: SINAN por dia (total do mês / dias) e média diária de cada mês;
      (2) projeção: casos importados acumulados desde o fim do observado,
          média (λ·t) e pior caso (quantil) a taxa constante.
    Retorna (fig, eixos, cursor), como `montar_figura`.
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clear()
    eixos = fig.subplots(2, 1)
    pc = res["pior_caso_importados"]

    d = res["diaria"][res["diaria"]["observado"]].copy()   # só o observado (SINAN)
    d.index = pd.to_datetime(d.pop("dia"))
    d = d.asfreq("D")   # periodo excluido vira NaN e interrompe a linha
    ax = eixos[0]
    linha, = ax.plot(d.index, d["valor"], color="royalblue", lw=1.2, label="Interpolacao linear do acumulado (casos por dia)")
    linha.rotulo_tooltip = ("Casos importados por dia", 1.0, 2, False)
    meio = pd.to_datetime([datetime.date(*ano_mes(v), 15) for v in res["x_mensal"]])
    pontos = ax.scatter(meio, res["y_mensal"] / _dias_do_mes(res["x_mensal"]), color="crimson",
                        s=14, alpha=0.7, zorder=5, label="Dados mensais (média/dia)")
    pontos.rotulo_tooltip = ("Média/dia do mês", 1.0, 2, False)
    for ini, fim in res.get("excluir_periodo") or []:
        ax.axvspan(pd.Timestamp(_dia_para_data(ini)), pd.Timestamp(_dia_para_data(fim)),
                   color="gray", alpha=0.08, zorder=0)
    ax.set_title(titulo or f"Casos importados de sarampo - observado (SINAN, total do mês / dias)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Casos por dia")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = eixos[1]
    dias = pd.to_datetime(pc["dias"])
    linha, = ax.plot(dias, pc["acumulado_media"], color="purple", lw=1.5,
                     label=f"Média λ·t ({formata_numero(pc['media'], 1)} casos)")
    linha.rotulo_tooltip = ("Acumulado (média)", 1.0, 1, False)
    linha, = ax.plot(dias, pc["acumulado_pior_caso"], color="black", lw=1.2, linestyle="--",
                     label=f"Pior caso {formata_quantil(pc['quantil'])} a taxa constante "
                           f"({formata_numero(pc['total'], 0)} casos; CSV diário e SEIHRVS)")
    linha.rotulo_tooltip = ("Acumulado (pior caso)", 1.0, 1, False)
    ax.set_title(f"Projeção pelo pior caso da Poisson: casos importados acumulados "
                 f"(λ = {formata_numero(pc['lambda_ano'], 2)} casos/ano)", fontsize=11)
    ax.set_ylabel("Casos importados acumulados")
    ax.set_xlabel("Data")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    cursor = mplcursors.cursor([a for e in eixos for a in e.lines + e.collections
                                if hasattr(a, "rotulo_tooltip")], hover=False)
    cursor.connect("add", _tooltip_mortalidade)
    return fig, eixos, cursor

def montar_figura_saldo(res, fig=None, titulo=None, figsize=(14, 8)):
    """
    Dois painéis com o mesmo eixo de datas: (1) as duas séries interpoladas
    (taxa diária) com a média diária de cada mês como pontos; (2) o saldo
    diário (diferença das duas), com a área integrada sombreada - verde acima
    de zero, vermelha abaixo - e o saldo mensal (média/dia) como pontos.
    Retorna (fig, eixos, cursor), como `montar_figura`.
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clear()
    eixos = fig.subplots(2, 1, sharex=True)

    # Grade diaria completa: dias do periodo excluido viram NaN e interrompem a linha
    d = res["diaria"].copy()
    d.index = pd.to_datetime(d.pop("dia"))
    d = d.asfreq("D")

    def _meio_do_mes(x_abs):
        return pd.to_datetime([datetime.date(*ano_mes(v), 15) for v in x_abs])

    # --- Painel 1: as duas séries interpoladas -------------------------------
    ax = eixos[0]
    for chave_res, coluna, rotulo, cor in (("res_a", res["coluna_a"], res["nome_a"], "darkorange"),
                                           ("res_b", res["coluna_b"], res["nome_b"], "royalblue")):
        linha, = ax.plot(d.index, d[coluna], color=cor, lw=1.2,
                         label=f"{rotulo} (taxa diária)")
        linha.rotulo_tooltip = (f"{rotulo}/dia", 1.0, 1, False)
        r = res[chave_res]
        pontos = ax.scatter(_meio_do_mes(r["x_mensal"]), r["y_mensal"] / _dias_do_mes(r["x_mensal"]),
                            color=cor, s=14, alpha=0.7, zorder=5, edgecolors="black", linewidths=0.3)
        pontos.rotulo_tooltip = (f"{rotulo} - média/dia do mês", 1.0, 1, False)
    ax.set_ylabel("Pessoas por dia")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Painel 2: saldo diário e área integrada -----------------------------
    ax = eixos[1]
    saldo = d["valor"]
    for cond, cor, rot in ((saldo >= 0, "seagreen", "Área positiva"),
                           (saldo <= 0, "indianred", "Área negativa")):
        ax.fill_between(d.index, saldo, 0, where=cond.to_numpy(), interpolate=True,
                        color=cor, alpha=0.25, zorder=1, label=rot)
    linha, = ax.plot(d.index, saldo, color="black", lw=1.0,
                     label="Saldo diário (diferença das interpolações)")
    linha.rotulo_tooltip = ("Saldo/dia", 1.0, 1, False)
    pontos = ax.scatter(_meio_do_mes(res["x_mensal"]), res["y_mensal"] / _dias_do_mes(res["x_mensal"]),
                        color="crimson", s=16, zorder=5, label="Saldo mensal (média/dia)")
    pontos.rotulo_tooltip = ("Saldo - média/dia do mês", 1.0, 1, False)
    ax.axhline(0, color="black", lw=0.8, alpha=0.6, zorder=1)
    ax.set_ylabel(textwrap.fill(f"{res['col_valor']} por dia", 30))
    ax.set_xlabel("Data")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    for ini, fim in res.get("excluir_periodo") or []:
        for eixo in eixos:
            eixo.axvspan(pd.Timestamp(_dia_para_data(ini)), pd.Timestamp(_dia_para_data(fim)),
                         color="gray", alpha=0.08, zorder=0)

    titulo_padrao = f"Saldo migratório - {res['col_valor']}"
    if res.get("excluir_periodo"):
        titulo_padrao += f"\n(sem o período da Covid-19: {descrever_periodos(res['excluir_periodo'])})"
    eixos[0].set_title(titulo or titulo_padrao, fontsize=13, fontweight="bold")
    fig.tight_layout()

    cursor = mplcursors.cursor([a for eixo in eixos for a in eixo.lines + eixo.collections
                                if hasattr(a, "rotulo_tooltip")], hover=False)
    cursor.connect("add", _tooltip_mortalidade)
    return fig, eixos, cursor

def _dia_para_data(x_absoluto):
    """Ano fracionário (datas reais) -> datetime.date."""
    return datetime.date.fromordinal(int(round(float(_ano_para_dia(x_absoluto)))))

def _tooltip_mortalidade(sel):
    """Data e valor do painel; Pi em notação científica (por pessoa por dia)."""
    rotulo, escala, casas, eh_pi = sel.artist.rotulo_tooltip
    data  = mdates.num2date(sel.target[0]).date()
    valor = f"{sel.target[1] / escala:.4e}" if eh_pi else formata_numero(sel.target[1], casas)
    sel.annotation.set_text(f"{data:%d/%m/%Y}\n{rotulo}: {valor}")


# =============================================================================
# 6. EXECUÇÃO COMO SCRIPT (a interface esta em interpolate_gui.py)
# =============================================================================

def _gravar_tabela(colunas, observado, nome, periodos, diretorio):
    """Grava `colunas` + tipo (observado/previsto) em '<nome>[sufixo].csv'; devolve o caminho."""
    periodos = periodos or []
    sufixo   = ("" if not periodos else
                "_sem_covid" if periodos == unir_periodos(PERIODO_COVID) else "_com_exclusao")
    os.makedirs(diretorio, exist_ok=True)
    caminho = os.path.join(diretorio, f"{nome}{sufixo}.csv")
    pd.DataFrame({**colunas, "tipo": np.where(observado, "observado", "previsto")}
                 ).to_csv(caminho, sep=SEPARADOR_CSV, index=False)
    return caminho

def salvar_nascimentos_diarios(res, diretorio=DIRETORIO_SAIDA_DIARIA):
    """
    Nascimentos por dia (o Λ do SEIHRVS) da análise do CSV de nascimentos, em
    'nascimentos_diarios.csv' (separador ';'): dia; nascimentos (pessoas/dia);
    tipo ('observado' ou 'previsto' pelo Holt-Winters). Com período excluído o
    nome ganha '_sem_covid' (ou '_com_exclusao'). Devolve o caminho.
    """
    d = res["diaria"]
    return _gravar_tabela({"dia": d["dia"], "nascimentos": d["valor"]}, d["observado"],
                          "nascimentos_diarios", res.get("excluir_periodo"), diretorio)

def salvar_importados_diarios(res, diretorio=DIRETORIO_SAIDA_DIARIA):
    """
    Casos importados de sarampo por dia (entrada diária do modelo), em
    'casos_importados_diarios.csv' (separador ';'): dia; casos_importados
    (casos/dia); tipo. 'observado' = total mensal do SINAN dividido entre os
    dias do mês; 'previsto' = pior caso da Poisson a taxa constante
    (pior_caso_importados), o ε do SEIHRVS. Com período excluído o nome ganha
    '_sem_covid' (ou '_com_exclusao').
    Devolve o caminho.
    """
    d = res["diaria"]
    return _gravar_tabela({"dia": d["dia"], "casos_importados": d["valor"]}, d["observado"],
                          "casos_importados_diarios", res.get("excluir_periodo"), diretorio)

def salvar_taxa_obitos_diaria(res, diretorio=DIRETORIO_SAIDA_DIARIA):
    """
    Taxa de óbito por dia (o π do SEIHRVS) da análise de mortalidade, em
    'taxa_obitos_diaria.csv' (separador ';'): dia; Pi (óbitos por pessoa por
    dia); obitos e populacao (de onde Pi = obitos / populacao sai); tipo.
    Com período excluído o nome ganha '_sem_covid' (ou '_com_exclusao').
    Devolve o caminho.
    """
    if res.get("tipo") != "mortalidade":
        raise ValueError("A taxa de óbito sai da análise de óbitos (analisar_mortalidade).")
    d = res["diaria"]
    return _gravar_tabela({"dia": d["dia"], "Pi": d["valor"],
                           "obitos": d["obitos"], "populacao": d["populacao"]},
                          d["observado"], "taxa_obitos_diaria",
                          res.get("excluir_periodo"), diretorio)

def main():
    res = analisar_csv(ARQUIVO_NASCIMENTOS,
                       separador=SEPARADOR_CSV,
                       anos_previsao=QUANTIDADE_ANOS_PREVISAO)
    print(resumo_texto(res))
    print(f"\nNascimentos diarios gravados em '{salvar_nascimentos_diarios(res)}'")
    montar_figura(res)
    plt.show()


if __name__ == "__main__":
    main()
