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
TAMANHO_GRAFICO = (11, 5.2)  # polegadas (largura, altura) de cada figura
DPI_GRAFICO     = 100        # pixels por polegada da figura

# A lista de CSVs cresce junto com o numero de arquivos ate este limite; passando
# dele, a coluna ganha altura fixa e barra de rolagem.
LIMITE_ROLAGEM   = 12
ALTURA_LINHA_CSV = 33   # px por checkbox (medido com Segoe UI 10)

# Cada CSV marcado vira uma janela de gráfico separada, guardada aqui por caminho
janelas_graficos = {}


# =============================================================================
# 1. JANELA PRINCIPAL (checkboxes dos CSVs + opções)
# =============================================================================

def descrever_csv(arquivo, separador):
    """Rótulo do checkbox: nome do arquivo + coluna de valor lida do cabeçalho."""
    nome = os.path.basename(arquivo)
    try:
        _, col_valor = itp.ler_colunas(arquivo, separador)
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


def montar_janela_principal(arquivos, separador):
    linhas = linhas_checkbox(arquivos, separador)
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


def abrir_janela_grafico(res):
    """Cria (ou recria) a janela de gráfico de um CSV e desenha a análise nela."""
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
        # R2 da extrapolacao REALMENTE plotada (curva roxa).
        [sg.Text(f"R² - {res['rotulo_modelo']} (curva plotada): "
                 f"{res['r2_tendencia']:.4f}", font=(FONT_FAMILY, FONT_SIZE, "bold"))],
        # Spline cubica INTERPOLA os pontos (passa por todos), entao R2=1 sempre:
        # e so uma confirmacao de que a interpolacao esta correta, nao uma metrica
        # de qualidade de ajuste.
        [sg.Text(f"R² - Spline cúbica (interpolação, sempre 1): {res['r2_spline']:.4f}")],
        [sg.Push(), sg.Button("Fechar", key=("-FECHAR-", arquivo))],
    ]
    sufixo = " (sem COVID-19)" if res.get("excluir_periodo") else ""
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
                                 "canvas": canvas, "cursor": cursor}
    return janela


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
    excluir   = itp.PERIODO_COVID if valores["-SEM-COVID-"] else None

    selecionados = csvs_marcados(valores, arquivos)
    if not selecionados:
        sg.popup("Marque ao menos um CSV para gerar gráficos.",
                 title="Nenhum arquivo selecionado", font=FONT_FAMILY)
        return

    for arquivo in selecionados:
        try:
            res = itp.analisar_csv(arquivo, separador=separador, anos_previsao=anos,
                                   excluir_periodo=excluir)
            abrir_janela_grafico(res)
        except Exception as erro:
            traceback.print_exc()
            sg.popup_error(f"Falha ao processar '{arquivo}':\n\n{erro}", font=FONT_FAMILY)


def main():
    sg.theme(TEMA)
    separador = itp.SEPARADOR_CSV
    arquivos  = itp.listar_csvs()
    janela    = montar_janela_principal(arquivos, separador)

    while True:
        win, evento, valores = sg.read_all_windows()

        if win is None:  # nenhuma janela aberta
            break

        if evento in (sg.WIN_CLOSED, "Sair") and win is janela:
            break

        # Eventos das janelas de gráfico
        if win is not janela:
            if evento == sg.WIN_CLOSED or (isinstance(evento, tuple) and evento[0] == "-FECHAR-"):
                alvo = next((a for a, d in janelas_graficos.items() if d["janela"] is win), None)
                if alvo:
                    fechar_janela_grafico(alvo)
                else:
                    win.close()
            continue

        if evento == "Marcar todos":
            for arq in arquivos:
                janela[("-CSV-", arq)].update(True)

        elif evento == "Limpar seleção":
            for arq in arquivos:
                janela[("-CSV-", arq)].update(False)

        elif evento == "Gerar gráficos":
            gerar_graficos(janela, valores, arquivos)

        elif evento == "Fechar gráficos":
            fechar_todos_graficos()

    fechar_todos_graficos()
    janela.close()


if __name__ == "__main__":
    main()
