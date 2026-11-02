"""
Leitos de UTI (adulto + pediátrica) livres no SUS, por mês, a partir dos CSVs
de exportar_datasus.py:

    dados/cnes/leitos_cnes_AAAA.csv        ->  leitos SUS de UTI (QT_SUS)
    dados/sih/dias_uti_sih_AAAA.csv         ->  dias de UTI usados por pacientes SUS
    dados/populacao/populacao_por_uf_AAAA.csv  ->  leitos livres por 10 mil habitantes

    leitos ocupados = dias de UTI da competência / dias do mês
    leitos livres   = leitos SUS - leitos ocupados
    ocupação        = leitos ocupados / leitos SUS

Os dias de UTI de cada AIH entram no mês de competência (mês em que a AIH foi
processada), não nos dias exatos em que o leito foi ocupado.

MARCA_UTI tem dois esquemas de código (tabelas CNV do SIH no FTP do DATASUS):
TAB_SIH_200308-200712 até dez/2007 e TAB_SIH a partir de jan/2008. Os códigos
que não correspondem a um único tipo de leito do CNES (ex.: "UTI tipo I ou não
utilizou", "mais de um tipo de UTI", transplante) ficam como não classificados;
por isso ocupação e leitos livres saem como intervalo: mínimo (sem os não
classificados) e máximo (com todos eles).

    python parametros/calcular_ocupacao_uti.py
    python parametros/calcular_ocupacao_uti.py --anos 2018 2019 2020
"""

import os
import argparse

import pandas as pd

from calcular_parametros_hospitalares import DIRETORIO_DADOS, ANOS, ler_csvs

# =============================================================================
# CONFIGURACOES
# =============================================================================

ARQUIVO_MENSAL_UF = os.path.join(DIRETORIO_DADOS, "resultados", "hospitalar", "ocupacao_uti_mensal_uf.csv")
ARQUIVO_BRASIL = os.path.join(DIRETORIO_DADOS, "resultados", "hospitalar", "ocupacao_uti_brasil.csv")

# Leitos de UTI no CNES (TP_LEITO = 3; tabela Leitos_compl2.cnv do TabNet)
TP_LEITO_COMPLEMENTAR = "3"
LEITOS_CNES = {"adulto": ["61", "74", "75", "76"], "pediatrica": ["62", "77", "78", "79"]}

# MARCA_UTI no SIH. Códigos que não aparecem aqui (neonatal, queimados,
# coronariana, COVID-19, doador) são de outros tipos de leito e não entram.
INICIO_ESQUEMA_NOVO = "200801"
MARCA_UTI_ANTIGO = {                       # TAB_SIH_200308-200712 / MARCAUTI.CNV
    "adulto": ["01", "02"],                # adulto nível II e III
    "pediatrica": ["07", "08"],            # pediátrica nível II e III
    "nao_classificado": ["00", "04", "05"] + [f"{c:02d}" for c in range(10, 100)],
                                           # tipo I ou não utilizou; transplante ped./adulto; ignorado
}
MARCA_UTI_NOVO = {                         # TAB_SIH / MARCAUTI.CNV
    "adulto": ["74", "75", "76"],
    "pediatrica": ["77", "78", "79"],
    "nao_classificado": ["00", "01"],      # 00 com dias de UTI (inconsistente); mais de um tipo
}

COLUNAS_CNES = ["UF", "COMPETEN", "TP_LEITO", "CODLEITO", "QT_SUS"]
COLUNAS_SIH_UTI = ["UF", "COMPETEN", "MARCA_UTI", "DIAS_UTI"]
COLUNAS_POPULACAO = ["UF", "POP"]


# =============================================================================
# 1. LEITURA
# =============================================================================

def carregar_leitos_sus(anos):
    """Leitos SUS de UTI adulto + pediátrica por competência x UF."""
    lt = ler_csvs("cnes", "leitos_cnes", anos, COLUNAS_CNES)
    codigos = [c for lista in LEITOS_CNES.values() for c in lista]
    lt = lt[(lt["TP_LEITO"] == TP_LEITO_COMPLEMENTAR) & lt["CODLEITO"].isin(codigos)]
    lt["QT_SUS"] = pd.to_numeric(lt["QT_SUS"], errors="coerce").fillna(0)
    return lt.groupby(["COMPETEN", "UF"])["QT_SUS"].sum().rename("leitos_sus")

def classificar_marca_uti(competen, marca):
    """Categoria de cada linha conforme o esquema de código do período."""
    categoria = pd.Series(pd.NA, index=marca.index, dtype="object")
    for esquema, mascara in [(MARCA_UTI_ANTIGO, competen < INICIO_ESQUEMA_NOVO),
                             (MARCA_UTI_NOVO, competen >= INICIO_ESQUEMA_NOVO)]:
        for nome, codigos in esquema.items():
            categoria[mascara & marca.isin(codigos)] = nome
    return categoria

def carregar_dias_uti(anos):
    """Dias de UTI (adulto, pediátrica e não classificados) por competência x UF."""
    uti = ler_csvs("sih", "dias_uti_sih", anos, COLUNAS_SIH_UTI)
    uti["DIAS_UTI"] = pd.to_numeric(uti["DIAS_UTI"], errors="coerce").fillna(0)
    uti["categoria"] = classificar_marca_uti(uti["COMPETEN"], uti["MARCA_UTI"])
    uti = uti.dropna(subset=["categoria"])
    dias = uti.pivot_table(index=["COMPETEN", "UF"], columns="categoria", values="DIAS_UTI",
                           aggfunc="sum", fill_value=0)
    return dias.reindex(columns=["adulto", "pediatrica", "nao_classificado"], fill_value=0) \
               .add_prefix("dias_")

def carregar_populacao(anos):
    """População por (ano, UF); anos sem estimativa usam o último disponível."""
    partes, ultima = [], None
    for ano in sorted(anos):
        try:
            pop = ler_csvs("populacao", "populacao_por_uf", [ano], COLUNAS_POPULACAO)
            ultima = pop.assign(POP=pd.to_numeric(pop["POP"]))
        except FileNotFoundError:
            print(f"  População {ano}: não publicada, usando a do último ano disponível")
        if ultima is not None:
            partes.append(ultima.assign(ano=ano))
    return pd.concat(partes).set_index(["ano", "UF"])["POP"]


# =============================================================================
# 2. CALCULO
# =============================================================================

def calcular(tabela):
    """Acrescenta ocupação e leitos livres (mínimo e máximo) a uma tabela com leitos e dias."""
    t = tabela.copy()
    competencia = pd.to_datetime(t.index.get_level_values("COMPETEN"), format="%Y%m")
    t["dias_no_mes"] = competencia.days_in_month
    classificados = t["dias_adulto"] + t["dias_pediatrica"]
    t["ocupados_min"] = classificados / t["dias_no_mes"]
    t["ocupados_max"] = (classificados + t["dias_nao_classificado"]) / t["dias_no_mes"]
    t["ocupacao_min"] = t["ocupados_min"] / t["leitos_sus"]
    t["ocupacao_max"] = t["ocupados_max"] / t["leitos_sus"]
    t["livres_min"] = t["leitos_sus"] - t["ocupados_max"]
    t["livres_max"] = t["leitos_sus"] - t["ocupados_min"]
    t["livres_por_10mil_min"] = t["livres_min"] / t["populacao"] * 1e4
    t["livres_por_10mil_max"] = t["livres_max"] / t["populacao"] * 1e4
    return t


# =============================================================================
# 3. EXECUCAO
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--anos", type=int, nargs="+", default=ANOS)
    args = parser.parse_args()

    print("Lendo leitos (CNES), dias de UTI (SIH) e população...")
    leitos = carregar_leitos_sus(args.anos)
    dias = carregar_dias_uti(args.anos)
    populacao = carregar_populacao(args.anos)

    por_uf = dias.join(leitos, how="inner")               # meses com as duas bases publicadas
    anos_idx = por_uf.index.get_level_values("COMPETEN").str[:4].astype(int)
    por_uf["populacao"] = populacao.reindex(
        list(zip(anos_idx, por_uf.index.get_level_values("UF")))).to_numpy()
    por_uf = calcular(por_uf)
    por_uf.to_csv(ARQUIVO_MENSAL_UF, sep=";")

    somas = ["dias_adulto", "dias_pediatrica", "dias_nao_classificado", "leitos_sus", "populacao"]
    brasil = calcular(por_uf[somas].groupby(level="COMPETEN").sum())
    brasil.to_csv(ARQUIVO_BRASIL, sep=";")

    anual = brasil.groupby(brasil.index.str[:4])[
        ["leitos_sus", "ocupacao_min", "ocupacao_max", "livres_min", "livres_max",
         "livres_por_10mil_min", "livres_por_10mil_max"]].mean()
    pd.set_option("display.width", 200)
    print()
    print("Média anual (Brasil, UTI adulto + pediátrica, SUS):")
    print(anual.to_string(float_format=lambda v: f"{v:,.2f}"))
    print(f"\nSérie mensal nacional: {ARQUIVO_BRASIL}")
    print(f"Série mensal por UF: {ARQUIVO_MENSAL_UF}")


if __name__ == "__main__":
    main()
