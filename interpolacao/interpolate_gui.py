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
        texto = rotulo + (f"   (falta {', '.join(faltando)})" if faltando else "")
        linhas.append([sg.Checkbox(texto, default=False, key=("-DIF-", rotulo),
                                   disabled=bool(faltando),
                                   font=(FONT_FAMILY, FONT_SIZE))])
    return linhas


def linha_mortalidade():
    """Checkbox da taxa diária de mortalidade; desabilitado se faltar algum CSV."""
    faltando = [os.path.basename(a) for a in (itp.ARQUIVO_OBITOS, itp.ARQUIVO_POPULACAO)
                if not os.path.isfile(a)]
    texto = ("Taxa diária de mortalidade (óbitos ÷ população IBGE, ambos histopolados)"
             + (f"   (falta {', '.join(faltando)})" if faltando else ""))
    return [[sg.Checkbox(texto, default=False, key="-MORT-", disabled=bool(faltando),
                         font=(FONT_FAMILY, FONT_SIZE))]]


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
        [sg.Frame("Séries derivadas (diferença mês a mês, só interpolação + integral)",
                  linhas_diferenca(),
                  expand_x=True)],
        [sg.Frame("Taxa de mortalidade (Pi do SEIHRVS, previsão pelo Holt-Winters dos óbitos)",
                  linha_mortalidade(),
                  expand_x=True)],
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
    ]
    if res["modelo_tendencia"] is not None:
        # R2 da extrapolacao REALMENTE plotada (curva roxa).
        layout.append([sg.Text(f"R² - {res['rotulo_modelo']} (curva plotada): "
                               f"{res['r2_tendencia']:.4f}",
                               font=(FONT_FAMILY, FONT_SIZE, "bold"))])
    if res.get("integral"):
        integ = res["integral"]
        layout += [
            [sg.Text(f"Integral da histospline: {itp.descrever_integral(integ)}",
                     font=(FONT_FAMILY, FONT_SIZE, "bold"))]
        ]
    if res.get("tipo") == "mortalidade":
        anual = res["mortalidade_anual"]
        layout += [
            [sg.Text(f"Pi médio por ano (por 100 mil hab./dia): " + "  ".join(
                f"{ano}: {itp.formata_numero(pi * 1e5, 3)}" for ano, pi in anual["Pi_medio"].items()),
                size=(130, None))],
            [sg.Text(f"População IBGE histopolada - maior |média do ano - valor anual| "
                     f"(deve ser ~0): {res['erro_populacao']:.2e}")],
        ]
    layout += [
        # A histospline conserva o total de cada mes por construcao: o erro so
        # confirma isso (fica no nivel do arredondamento), nao mede ajuste.
        [sg.Text(f"Histospline{' dos óbitos' if res.get('tipo') == 'mortalidade' else ''}"
                 f" - maior erro do total mensal (deve ser ~0): "
                 f"{res['erro_conservacao']:.2e}")],
        [sg.Button("Gravar série diária (CSV)", key=("-SALVAR-", arquivo)),
         sg.Push(), sg.Button("Fechar", key=("-FECHAR-", arquivo))],
    ]
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


def gravar_serie_diaria(arquivo):
    """Grava a série diária (observado + previsão) da janela e avisa onde ficou."""
    res = janelas_graficos[arquivo]["res"]
    try:
        caminho = os.path.abspath(itp.salvar_serie_diaria(res))
    except Exception as erro:
        traceback.print_exc()
        sg.popup_error(f"Falha ao gravar a série diária:\n\n{erro}", font=FONTE_PADRAO)
        return
    diaria = res["diaria"]
    previstos = int((~diaria["observado"]).sum())
    sg.popup(f"Série diária gravada em:\n{caminho}\n\n"
             f"{len(diaria)} dias ({len(diaria) - previstos} observados, "
             f"{previstos} previstos)",
             title="Série diária", font=FONTE_PADRAO)


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
    diferencas   = [r for r in SERIES_DIFERENCA if valores.get(("-DIF-", r))]
    mortalidade  = bool(valores.get("-MORT-"))
    if not selecionados and not diferencas and not mortalidade:
        sg.popup("Marque ao menos um CSV ou série derivada para gerar gráficos.",
                 title="Nenhum arquivo selecionado", font=FONTE_PADRAO)
        return

    for arquivo in selecionados:
        try:
            res = itp.analisar_csv(arquivo, separador=separador, anos_previsao=anos,
                                   excluir_periodo=excluir)
            abrir_janela_grafico(res)
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

    if mortalidade:
        try:
            abrir_janela_grafico(itp.analisar_mortalidade(anos_previsao=anos, excluir_periodo=excluir))
        except Exception as erro:
            traceback.print_exc()
            sg.popup_error(f"Falha ao calcular a taxa de mortalidade:\n\n{erro}", font=FONTE_PADRAO)


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

        if evento in (sg.WIN_CLOSED, "Sair") and win is janela:
            break

        # Eventos das janelas de gráfico
        if win is not janela:
            if isinstance(evento, tuple) and evento[0] == "-SALVAR-":
                gravar_serie_diaria(evento[1])
            elif evento == sg.WIN_CLOSED or (isinstance(evento, tuple) and evento[0] == "-FECHAR-"):
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
