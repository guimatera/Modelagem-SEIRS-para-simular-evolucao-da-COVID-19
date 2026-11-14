import os
import sys
import traceback
import PySimpleGUI as sg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import interpolate as itp

# Permite rodar de qualquer diretorio: acha o interpolate.py ao lado (ou acima)
_AQUI = os.path.dirname(os.path.abspath(__file__))
for _pasta in (_AQUI, os.path.dirname(_AQUI)):
    if _pasta not in sys.path:
        sys.path.insert(0, _pasta)

# =============================================================================
# CONFIGURACOES DA INTERFACE
# =============================================================================

TEMA           = "SystemDefault"
FONT_FAMILY   = "Segoe UI"
FONT_SIZE      = 10
FONTE_PADRAO   = (FONT_FAMILY, FONT_SIZE)
TAMANHO_GRAFICO = (11, 5.2)  # polegadas (largura, altura) de cada figura
DPI_GRAFICO     = 100        # pixels por polegada da figura

# A lista de CSVs cresce junto com o numero de arquivos ate este limite; passando
# dele, a coluna ganha altura fixa e barra de rolagem.
LIMITE_ROLAGEM   = 12
ALTURA_LINHA_CSV = 33   # px por checkbox (medido com Segoe UI 10)

# Cada CSV marcado vira uma janela de gráfico separada, guardada aqui por caminho
janelas_graficos = {}

# Séries derivadas: diferença mês a mês (A - B) entre dois CSVs do diretório de
# dados. Chave = rótulo do checkbox; o nome também identifica a janela do gráfico.
SERIES_DIFERENCA = {
    "Emigração - Imigração (saídas - entradas)": (
        "emigracao_mensal_brasil.csv", "imigracao_mensal_brasil.csv", "emigracao-imigracao"),
}


# =============================================================================
# 1. JANELA PRINCIPAL (checkboxes dos CSVs + opções)
# =============================================================================

def descrever_csv(arquivo, separador):
    """Rótulo do checkbox: nome do arquivo + coluna de valor lida do cabeçalho."""
    nome = os.path.basename(arquivo)
    try:
        _, col_valor = itp.ler_colunas(arquivo, separador)
        if itp.eh_arquivo_obitos(arquivo):
            if not os.path.isfile(itp.ARQUIVO_POPULACAO):
                return (f"{nome}   ({col_valor} - falta "
                        f"{os.path.basename(itp.ARQUIVO_POPULACAO)} para a taxa de mortalidade)")
            return f"{nome}   ({col_valor})"
        return f"{nome}   ({col_valor})"
    except Exception:
        return f"{nome}   (cabecalho nao lido)"


def linhas_checkbox(arquivos, separador):
    """Uma linha de checkbox por CSV encontrado no diretório de dados."""
    if not arquivos:
        return [[sg.Text(f"Nenhum CSV encontrado em:\n{itp.DIRETORIO_DADOS}",
                         text_color="firebrick")]]
    return [
        [sg.Checkbox(descrever_csv(arq, separador), default=(i == 0),
                     key=("-CSV-", arq), font=(FONT_FAMILY, FONT_SIZE))]
        for i, arq in enumerate(arquivos)
    ]


def linhas_diferenca():
    """Um checkbox por série derivada; desabilitado se faltar algum dos CSVs."""
    linhas = []
    for rotulo, (arq_a, arq_b, _) in SERIES_DIFERENCA.items():
        faltando = [a for a in (arq_a, arq_b)
                    if not os.path.isfile(os.path.join(itp.DIRETORIO_DADOS, a))]
        texto = (f"Saldo migratório: {rotulo}"
                 + (f"   (falta {', '.join(faltando)})" if faltando else ""))
        linhas.append([sg.Checkbox(texto, default=False, key=("-DIF-", rotulo),
                                   disabled=bool(faltando),
                                   font=(FONT_FAMILY, FONT_SIZE))])
    return linhas


def montar_janela_principal(arquivos, separador):
    # O saldo migratório entra na mesma lista dos CSVs, logo abaixo deles
    linhas = linhas_checkbox(arquivos, separador) + linhas_diferenca()
    # Sem altura fixa a coluna se ajusta ao conteudo e nenhum CSV fica cortado,
    # seja qual for a fonte. A altura fixa anterior (30 px por linha) era menor
    # que a linha real (33 px), e o ultimo arquivo sumia sem barra de rolagem.
    rolar = len(linhas) > LIMITE_ROLAGEM
    coluna_csvs = sg.Column(
        linhas,
        scrollable=rolar, vertical_scroll_only=True,
        size=(520, ALTURA_LINHA_CSV * LIMITE_ROLAGEM) if rolar else (None, None),
        expand_x=True,
    )

    opcoes = [
        [sg.Checkbox("Extrapolar com Holt-Winters", default=False, key="-HW-",
                     enable_events=True)],
        [sg.Text("Anos de previsão:", size=(18, 1)),
         sg.Spin([i for i in range(1, 51)], initial_value=itp.QUANTIDADE_ANOS_PREVISAO,
                 key="-ANOS-", size=(5, 1))],
        [sg.Checkbox("Excluir o período da COVID-19",
                     default=False, key="-SEM-COVID-")]
    ]

    layout = [
        [sg.Text("Interpolação e Extrapolação de Séries Temporais",
                 font=(FONT_FAMILY, FONT_SIZE))], 
        [sg.Frame("Arquivos CSV", [[coluna_csvs]], expand_x=True)],
        [sg.Button("Marcar todos"), sg.Button("Limpar seleção")],
        [sg.Frame("Opções", opcoes, expand_x=True)],
        [sg.Button("Gerar gráficos", bind_return_key=True),
         sg.Button("Fechar gráficos"), sg.Push(), sg.Button("Sair")]
    ]

    return sg.Window("Interpolação de Séries Temporais", layout,
                     font=(FONT_FAMILY, FONT_SIZE), finalize=True)

# =============================================================================
# 2. JANELA DE GRÁFICO (uma por CSV, com figura matplotlib embutida)
# =============================================================================

class BarraFerramentas(NavigationToolbar2Tk):
    """Toolbar do matplotlib (zoom/pan/salvar) empacotada dentro da janela."""
    def __init__(self, canvas, pai):
        super().__init__(canvas, pai, pack_toolbar=False)


def _texto_ajuste(nome, r):
    """'<nome>: R² Holt-Winters = 0,9501  |  RMSE = 6.789 nascimentos/mês'."""
    unidade = r["col_valor"].split(" no ")[0].lower()   # "Nascimentos no Brasil" -> "nascimentos"
    return (f"{nome}R² do Holt-Winters = {itp.formata_numero(r['r2_tendencia'], 4)}  |  "
            f"RMSE da série temporal = {itp.formata_numero(r['rmse_tendencia'])} {unidade}/mês")


def linhas_consideracoes(res):
    """
    Considerações abaixo do gráfico:
      saldo migratório -> só a integral do saldo diário;
      demais séries    -> R² e RMSE do Holt-Winters sobre os totais mensais
                          observados (nada, se a extrapolação estiver desligada).
    Na mortalidade aparecem os dois ajustes que a geram: óbitos e nascimentos.
    """
    negrito = (FONT_FAMILY, FONT_SIZE, "bold")
    if res.get("tipo") == "saldo":
        return [[sg.Text(f"Integral do saldo migratório: {itp.descrever_integral(res['integral'])}",
                         font=negrito)]]
    if res.get("tipo") == "mortalidade":
        ajustes = [("Óbitos - ", res["res_obitos"]), ("Nascimentos - ", res["res_nascimentos"])]
    else:
        ajustes = [("", res)]
    return [[sg.Text(_texto_ajuste(nome, r), font=negrito)]
            for nome, r in ajustes if r.get("rmse_tendencia") is not None]


def abrir_janela_grafico(res, gravados=()):
    """
    Cria (ou recria) a janela de gráfico de um CSV e desenha a análise nela.
    `gravados`: caminhos dos CSVs já gravados para esta análise (mostrados na janela).
    """
    arquivo = res["arquivo"]
    fechar_janela_grafico(arquivo)

    # O sg.Canvas nao cresce sozinho para caber a figura: precisa do tamanho em
    # pixels (polegadas * dpi), senao o grafico fica reduzido a alguns pixels
    largura_px = int(TAMANHO_GRAFICO[0] * DPI_GRAFICO)
    altura_px  = int(TAMANHO_GRAFICO[1] * DPI_GRAFICO)

    layout = [
        [sg.Text(res["col_valor"], font=(FONT_FAMILY, FONT_SIZE+1))],
        [sg.Canvas(key="-TOOLBAR-", size=(largura_px, 40))],
        [sg.Canvas(key="-CANVAS-", size=(largura_px, altura_px))],
    ]
    layout += linhas_consideracoes(res)
    periodos = res.get("excluir_periodo")
    sufixo   = f" (sem {itp.descrever_periodos(periodos)})" if periodos else ""
    janela = sg.Window(f"Gráfico - {os.path.basename(arquivo)}{sufixo}", layout,
                       font=(FONT_FAMILY, FONT_SIZE), finalize=True, resizable=True)

    # O canvas do Tk precisa estar ligado à figura ANTES de desenhar, senão o
    # tooltip do mplcursors fica preso ao canvas provisório e nao responde
    fig    = Figure(figsize=TAMANHO_GRAFICO, dpi=DPI_GRAFICO)
    canvas = FigureCanvasTkAgg(fig, janela["-CANVAS-"].TKCanvas)
    _, _, cursor = itp.montar_figura(res, fig=fig)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

    barra = BarraFerramentas(canvas, janela["-TOOLBAR-"].TKCanvas)
    barra.update()
    barra.pack(side="left", fill="x")

    janela.refresh()
    janela.bring_to_front()

    # Guarda as referências: sem elas o tooltip (mplcursors) e o canvas morrem
    janelas_graficos[arquivo] = {"janela": janela, "fig": fig,
                                 "canvas": canvas, "cursor": cursor, "res": res}
    return janela


def gravar_csvs(res):
    """
    Grava o CSV diário de uma análise e devolve os caminhos (um por análise):
      nascimentos -> nascimentos_diarios.csv (nascimentos por dia)
      óbitos      -> taxa_obitos_diaria.csv  (Pi = óbitos por pessoa por dia)
    Outras séries (ex.: saldo migratório) não gravam nada.
    """
    if res.get("tipo") == "mortalidade":
        caminhos = [itp.salvar_taxa_obitos_diaria(res)]
    elif itp.eh_arquivo_nascimentos(res["arquivo"]):
        caminhos = [itp.salvar_nascimentos_diarios(res)]
    else:
        caminhos = []
    return [os.path.abspath(c) for c in caminhos]


def fechar_janela_grafico(arquivo):
    dados = janelas_graficos.pop(arquivo, None)
    if dados:
        dados["janela"].close()


def fechar_todos_graficos():
    for arquivo in list(janelas_graficos):
        fechar_janela_grafico(arquivo)


# =============================================================================
# 3. LOOP PRINCIPAL
# =============================================================================

def csvs_marcados(valores, arquivos):
    return [arq for arq in arquivos if valores.get(("-CSV-", arq))]


def gerar_graficos(janela, valores, arquivos):
    separador = ";"
    anos      = int(valores["-ANOS-"])
    extrapolar = bool(valores["-HW-"])
    excluir   = itp.PERIODO_COVID if valores["-SEM-COVID-"] else None

    selecionados = csvs_marcados(valores, arquivos)
    diferencas   = [r for r in SERIES_DIFERENCA if valores.get(("-DIF-", r))]
    if not selecionados and not diferencas:
        sg.popup("Marque ao menos um CSV ou série derivada para gerar gráficos.",
                 title="Nenhum arquivo selecionado", font=FONTE_PADRAO)
        return

    for arquivo in selecionados:
        try:
            # Óbitos: histopolação dos óbitos e da população IBGE e a taxa
            # diária de mortalidade total do Brasil (Pi = óbitos/dia ÷ população)
            if itp.eh_arquivo_obitos(arquivo):
                res = itp.analisar_mortalidade(anos_previsao=anos, excluir_periodo=excluir,
                                               arquivo_obitos=arquivo, extrapolar=extrapolar)
            else:
                res = itp.analisar_csv(arquivo, separador=separador, anos_previsao=anos,
                                       excluir_periodo=excluir, extrapolar=extrapolar)
            abrir_janela_grafico(res, gravados=gravar_csvs(res))
        except Exception as erro:
            traceback.print_exc()
            sg.popup_error(f"Falha ao processar '{arquivo}':\n\n{erro}", font=FONTE_PADRAO)

    for rotulo in diferencas:
        arq_a, arq_b, nome = SERIES_DIFERENCA[rotulo]
        try:
            res = itp.analisar_diferenca(os.path.join(itp.DIRETORIO_DADOS, arq_a),
                                         os.path.join(itp.DIRETORIO_DADOS, arq_b),
                                         separador=separador,
                                         excluir_periodo=excluir, nome=nome)
            abrir_janela_grafico(res)
        except Exception as erro:
            traceback.print_exc()
            sg.popup_error(f"Falha ao processar '{rotulo}':\n\n{erro}", font=FONTE_PADRAO)


def main():
    sg.theme(TEMA)
    sg.set_options(font=FONTE_PADRAO)
    separador = itp.SEPARADOR_CSV
    arquivos  = itp.listar_csvs()
    janela    = montar_janela_principal(arquivos, separador)

    while True:
        win, evento, valores = sg.read_all_windows()
        if win is None:  # nenhuma janela aberta
            break

        # Sair (ou fechar a janela principal) encerra tudo, gráficos inclusive
        if evento in (sg.WIN_CLOSED, "Sair") and win is janela:
            break

        # Janelas de gráfico: fecham pelo X da própria janela
        if win is not janela:
            if evento == sg.WIN_CLOSED:
                alvo = next((a for a, d in janelas_graficos.items() if d["janela"] is win), None)
                if alvo:
                    fechar_janela_grafico(alvo)
                else:
                    win.close()
            continue

        if evento == "-HW-":
            janela["-ANOS-"].update(disabled=not valores["-HW-"])

        elif evento in ("Marcar todos", "Limpar seleção"):
            marcar = evento == "Marcar todos"
            for arq in arquivos:
                janela[("-CSV-", arq)].update(marcar)
            for rotulo in SERIES_DIFERENCA:
                caixa = janela[("-DIF-", rotulo)]
                if not caixa.Disabled:   # saldo sem os CSVs de origem fica desmarcado
                    caixa.update(marcar)

        elif evento == "Gerar gráficos":
            gerar_graficos(janela, valores, arquivos)

        elif evento == "Fechar gráficos":
            fechar_todos_graficos()

    fechar_todos_graficos()
    janela.close()


if __name__ == "__main__":
    main()
