"""
Capacidade de leitos de UTI (parâmetro ICU do modelo) a partir dos CSVs de
exportar_datasus.py:

    dados/cnes/leitos_cnes_AAAA.csv         ->  leitos de UTI por tipo, UF e mês
    dados/populacao/populacao_por_uf_AAAA.csv   ->  população por UF e ano

O modelo (SEIHRVS.py, campo '-uti-') recebe a capacidade em leitos por 10 mil
habitantes. Entram no ICU do modelo as UTIs adulto e pediátrica: foram as usadas
por 112 das 114 internações de sarampo em UTI no SIH (2007-2026); UTI neonatal,
coronariana, de queimados e as exclusivas de COVID-19 ficam de fora do total.
Os leitos livres (descontada a ocupação por pacientes SUS) estão em
calcular_ocupacao_uti.py.

    python parametros/calcular_leitos_uti.py
    python parametros/calcular_leitos_uti.py --anos 2018 2019 2020
"""

import os
import argparse

import pandas as pd

from calcular_parametros_hospitalares import ANOS, DIRETORIO_DADOS, ler_csvs

# =============================================================================
# CONFIGURACOES
# =============================================================================

ARQUIVO_MENSAL_UF = os.path.join(DIRETORIO_DADOS, "resultados", "leitos", "leitos_uti_mensal_uf.csv")
ARQUIVO_BRASIL = os.path.join(DIRETORIO_DADOS, "resultados", "leitos", "leitos_uti_brasil.csv")

# Leitos complementares (TP_LEITO = 3). Códigos conforme a tabela de conversão
# do TabNet (cnes/cnv/Leitos_compl2.cnv); 61-63 são a classificação antiga,
# anterior à separação em tipos I, II e III.
TP_LEITO_COMPLEMENTAR = "3"
CATEGORIAS_UTI = {
    "adulto":     ["61", "74", "75", "76"],
    "pediatrica": ["62", "77", "78", "79"],
    "neonatal":   ["63", "80", "81", "82"],
    "covid":      ["51", "52"],             # UTI II adulto/pediátrica COVID-19 (2020-2022)
    "outras":     ["83", "85", "86"],       # queimados e coronarianas
}
CATEGORIAS_MODELO = ["adulto", "pediatrica"]

COLUNAS_CNES = ["UF", "COMPETEN", "TP_LEITO", "CODLEITO", "QT_EXIST", "QT_SUS"]
COLUNAS_POPULACAO = ["UF", "POP"]


# =============================================================================
# 1. LEITURA
# =============================================================================

def carregar_leitos(anos):
    """Leitos de UTI por competência x UF x categoria."""
    lt = ler_csvs("cnes", "leitos_cnes", anos, COLUNAS_CNES)
    lt = lt[lt["TP_LEITO"] == TP_LEITO_COMPLEMENTAR]

    categoria = {codigo: nome for nome, codigos in CATEGORIAS_UTI.items() for codigo in codigos}
    lt = lt.assign(categoria=lt["CODLEITO"].map(categoria)).dropna(subset=["categoria"])
    for col in ["QT_EXIST", "QT_SUS"]:
        lt[col] = pd.to_numeric(lt[col], errors="coerce").fillna(0).astype(int)
    lt["competencia"] = lt["COMPETEN"].str[:4] + "-" + lt["COMPETEN"].str[4:6]

    return (lt.groupby(["competencia", "UF", "categoria"], as_index=False)[["QT_EXIST", "QT_SUS"]].sum()
              .rename(columns={"QT_EXIST": "existentes", "QT_SUS": "sus"}))

def carregar_populacao(anos):
    """
    População por ano (linhas) e UF (colunas). Anos ainda sem estimativa
    publicada usam a do último ano disponível (avisando).
    """
    series, ultima, ultimo_ano = {}, None, None
    for ano in sorted(anos):
        try:
            pop = ler_csvs("populacao", "populacao_por_uf", [ano], COLUNAS_POPULACAO)
            ultima, ultimo_ano = pd.to_numeric(pop.set_index("UF")["POP"]), ano
        except FileNotFoundError:
            print(f"  População {ano}: não publicada, usando a de {ultimo_ano}")
        series[ano] = ultima
    return pd.DataFrame(series).T.rename_axis("ano")


# =============================================================================
# 2. EXECUCAO
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--anos", type=int, nargs="+", default=ANOS)
    args = parser.parse_args()

    print("Lendo leitos do CNES...")
    leitos = carregar_leitos(args.anos)
    leitos.to_csv(ARQUIVO_MENSAL_UF, sep=";", index=False)

    print("Lendo população...")
    populacao = carregar_populacao(args.anos)

    # Série nacional mensal: leitos por categoria + total do modelo por 10 mil hab.
    brasil = leitos.pivot_table(index="competencia", columns="categoria",
                                values=["existentes", "sus"], aggfunc="sum", fill_value=0)
    brasil.columns = [f"{cat}_{tipo}" for tipo, cat in brasil.columns]
    brasil["uti_modelo_existentes"] = sum(brasil[f"{c}_existentes"] for c in CATEGORIAS_MODELO)
    brasil["uti_modelo_sus"] = sum(brasil[f"{c}_sus"] for c in CATEGORIAS_MODELO)
    brasil["populacao"] = [populacao.loc[int(c[:4])].sum() for c in brasil.index]
    brasil["uti_por_10mil_existentes"] = brasil["uti_modelo_existentes"] / brasil["populacao"] * 1e4
    brasil["uti_por_10mil_sus"] = brasil["uti_modelo_sus"] / brasil["populacao"] * 1e4
    brasil.to_csv(ARQUIVO_BRASIL, sep=";")

    anual = brasil.groupby(brasil.index.str[:4])[
        ["uti_modelo_existentes", "uti_modelo_sus", "uti_por_10mil_existentes", "uti_por_10mil_sus"]].mean()
    print()
    print("Média anual (Brasil, UTI adulto + pediátrica):")
    print(anual.to_string(float_format=lambda v: f"{v:,.2f}"))
    print(f"\nSérie mensal nacional: {ARQUIVO_BRASIL}")
    print(f"Série mensal por UF e categoria: {ARQUIVO_MENSAL_UF}")


if __name__ == "__main__":
    main()
