"""
Baixa os microdados públicos do DATASUS (FTP ftp.datasus.gov.br, sem login) e
os grava em CSV, um arquivo por ano, em dados/ (raiz do repositório):

    sinan/notificacoes_exantematicas_sinan_AAAA.csv  SINAN - Doenças exantemáticas (EXANBRaa.dbc):
                                        todas as notificações, por ano de notificação
    sih/internacoes_sarampo_sih_AAAA.csv            SIH - AIH reduzida (RDufaamm.dbc): AIHs com CID B05 no
                                        diagnóstico principal ou secundário, por ano de competência
    sih/dias_uti_sih_AAAA.csv            SIH - AIH reduzida (RDufaamm.dbc): dias de UTI de todas
                                        as AIHs, somados por competência x UF x tipo de UTI
    cnes/leitos_cnes_AAAA.csv           CNES - Leitos (LTufaamm.dbc): leitos somados por
                                        competência x UF x tipo de leito
    populacao/populacao_por_uf_AAAA.csv     IBGE/SVS - População estimada (POPSBRaa.zip), por UF

Só vão para o CSV as colunas listadas em COLUNAS_SINAN e COLUNAS_SIH (definidas
em calcular_parametros_hospitalares.py, que as usa) e em COLUNAS_CNES e
COLUNAS_POPULACAO e COLUNAS_SIH_UTI (abaixo). Para incluir uma coluna nova,
acrescente-a na lista e rode este script de novo: SINAN e população são relidos
do cache; SIH e CNES precisam ser baixados de novo (apague os recortes em
dados/cache_datasus/sih_recortes_espec, dados/cache_datasus/sih_uti e dados/cache_datasus/cnes_leitos).
O recorte do SIH inclui ESPEC (especialidade do leito da AIH) e CNES (hospital); a
pasta antiga sih_recortes, sem essas colunas, não é mais usada e pode ser apagada.

Os scripts de cálculo leem só esses CSVs. A descompactação dos .dbc é feita pelo
módulo dbc.py, em Python puro (qualquer versão). Se o pacote datasus-dbc estiver
instalado (binário para Windows só no Python 3.12), ele é usado no lugar, por ser
mais rápido; o resultado é o mesmo, byte a byte.
    python parametros/exportar_datasus.py
    python parametros/exportar_datasus.py --bases sinan cnes --anos 2018 2019
"""

import os
import time
import struct
import ftplib
import zipfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from dbc import descompactar
from calcular_parametros_hospitalares import (RAIZ, DIRETORIO_DADOS, ANOS, TODAS_UFS, CID, COLUNAS_DIAG,
                                              COLUNAS_SINAN, COLUNAS_SIH)

# =============================================================================
# CONFIGURACOES
# =============================================================================

BASES = ["sinan", "sih", "sih_uti", "cnes", "populacao"]

COLUNAS_CNES = ["UF", "COMPETEN", "TP_LEITO", "CODLEITO", "QT_EXIST", "QT_SUS"]
COLUNAS_POPULACAO = ["UF", "POP"]
COLUNAS_SIH_UTI = ["UF", "COMPETEN", "MARCA_UTI", "DIAS_UTI"]

# .dbc baixados e recortes por arquivo. Cada .dbc do SIH e do CNES é apagado
# depois de lido; o recorte fica e permite retomar uma execução interrompida.
DIRETORIO_CACHE = os.path.join(DIRETORIO_DADOS, "cache_datasus")

PROCESSOS = 4                                       # arquivos do SIH/CNES em paralelo

FTP_HOST = "ftp.datasus.gov.br"
FTP_SINAN = ["/dissemin/publicos/SINAN/DADOS/FINAIS/",
             "/dissemin/publicos/SINAN/DADOS/PRELIM/"]
FTP_SIH = ["/dissemin/publicos/SIHSUS/200801_/Dados/",
           "/dissemin/publicos/SIHSUS/199201_200712/Dados/"]   # até dez/2007
FTP_CNES_LT = ["/dissemin/publicos/CNES/200508_/Dados/LT/"]
FTP_POP = ["/dissemin/publicos/IBGE/POPSVS/"]

# Código IBGE da UF (2 primeiros dígitos do município) -> sigla
UF_IBGE = {"11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
           "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
           "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
           "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF"}


# =============================================================================
# 1. DOWNLOAD E LEITURA DOS .DBC
# =============================================================================

def baixar(nome, pastas, tentativas=3):
    """
    Baixa `nome` da primeira pasta do FTP que o tiver, guardando em cache.
    Devolve o caminho local, ou None se o arquivo não existir em nenhuma pasta.
    """
    os.makedirs(DIRETORIO_CACHE, exist_ok=True)
    local = os.path.join(DIRETORIO_CACHE, nome)
    if os.path.exists(local):
        return local

    for tentativa in range(1, tentativas + 1):
        try:
            ftp = ftplib.FTP(FTP_HOST, timeout=120)
            ftp.login()
            for pasta in pastas:
                try:
                    ftp.size(pasta + nome)
                except ftplib.error_perm:
                    continue                          # não está nesta pasta
                parcial = local + ".parcial"
                with open(parcial, "wb") as saida:
                    ftp.retrbinary("RETR " + pasta + nome, saida.write)
                os.replace(parcial, local)            # só entra no cache se completo
                ftp.quit()
                return local
            ftp.quit()
            return None
        except ftplib.all_errors as erro:
            print(f"    falha ao baixar {nome} ({erro}), tentativa {tentativa}/{tentativas}")
            time.sleep(5 * tentativa)
    raise RuntimeError(f"não foi possível baixar {nome}")

def ler_dbc(caminho, colunas=None, filtro=None):
    """
    Lê um .dbc do DATASUS como DataFrame de texto. `colunas=None` traz todas;
    as pedidas que não existem no arquivo são ignoradas.

    O DBF tem registros de tamanho fixo, então as colunas são fatiadas direto do
    buffer com numpy. `filtro(fatias) -> máscara` permite descartar linhas antes
    de decodificar, o que deixa a leitura do SIH (centenas de milhares de AIHs
    por arquivo, poucas de sarampo) rápida.
    """
    with open(caminho, "rb") as f:
        return ler_dbf(descompactar(f.read()), colunas, filtro)

def ler_dbf(dbf, colunas=None, filtro=None):
    """Lê o conteúdo (bytes) de um DBF já descompactado; ver ler_dbc."""
    n_registros = struct.unpack("<I", dbf[4:8])[0]
    tam_cabecalho = struct.unpack("<H", dbf[8:10])[0]
    tam_registro = struct.unpack("<H", dbf[10:12])[0]

    # Descritores de campo: 32 bytes cada, terminados por 0x0D. O byte 0 de cada
    # registro é a marca de exclusão, por isso o primeiro campo começa em 1.
    campos, i, deslocamento = {}, 32, 1
    while dbf[i] != 0x0D:
        nome = dbf[i:i + 11].split(b"\0")[0].decode("latin1").upper()  # POPSBR25 vem em minúsculas
        tamanho = dbf[i + 16]
        campos[nome] = (deslocamento, tamanho)
        deslocamento += tamanho
        i += 32

    registros = np.frombuffer(dbf, dtype=np.uint8, count=n_registros * tam_registro,
                              offset=tam_cabecalho).reshape(n_registros, tam_registro)

    def fatia(nome):
        ini, tam = campos[nome]
        return np.ascontiguousarray(registros[:, ini:ini + tam]).view(f"S{tam}").ravel()

    presentes = list(campos) if colunas is None else [c for c in colunas if c in campos]
    if filtro is not None:
        mascara = filtro({c: fatia(c) for c in presentes})
        registros = registros[mascara]

    return pd.DataFrame({c: pd.Series(fatia(c)).str.decode("latin1").str.strip()
                         for c in presentes})

def gravar_csv(df, caminho, colunas=None):
    """
    Grava no mesmo formato dos CSVs de interpolacao/dados (separador ';'),
    só com as `colunas` pedidas, na ordem dada (as ausentes saem vazias).
    """
    if colunas is not None:
        df = df.reindex(columns=colunas, fill_value="")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    df.to_csv(caminho + ".parcial", sep=";", index=False, encoding="utf-8")
    os.replace(caminho + ".parcial", caminho)

def processar_em_paralelo(funcao, tarefas, rotulo):
    """Roda `funcao(*tarefa)` para cada tarefa e devolve {tarefa: resultado}."""
    resultados, inicio = {}, time.time()
    with ProcessPoolExecutor(max_workers=PROCESSOS) as executor:
        futuros = {executor.submit(funcao, *t): t for t in tarefas}
        for k, futuro in enumerate(as_completed(futuros), start=1):
            resultados[futuros[futuro]] = futuro.result()
            if k % 250 == 0 or k == len(tarefas):
                print(f"  {rotulo}: {k}/{len(tarefas)} arquivos ({(time.time() - inicio) / 60:.0f} min)",
                      flush=True)
    return resultados

def recorte_em_cache(pasta, nome_dbc, pastas_ftp, extrair):
    """
    Devolve o recorte de um .dbc: do cache, se já existe; senão baixa, aplica
    `extrair(caminho) -> DataFrame`, guarda em cache e apaga o .dbc.
    None se o arquivo não foi publicado.
    """
    recorte = os.path.join(DIRETORIO_CACHE, pasta, nome_dbc.replace(".dbc", ".csv"))
    if os.path.exists(recorte):
        return pd.read_csv(recorte, sep=";", dtype=str, keep_default_na=False)
    caminho = baixar(nome_dbc, pastas_ftp)
    if caminho is None:
        return None
    df = extrair(caminho)
    gravar_csv(df, recorte)
    os.remove(caminho)
    return df


# =============================================================================
# 2. EXPORTACAO POR BASE
# =============================================================================

def exportar_sinan(anos):
    """Um CSV por ano de notificação, com todas as notificações."""
    for ano in anos:
        nome = f"EXANBR{ano % 100:02d}.dbc"
        caminho = baixar(nome, FTP_SINAN)
        if caminho is None:
            print(f"  SINAN {ano}: não publicado")
            continue
        df = ler_dbc(caminho, COLUNAS_SINAN)
        gravar_csv(df, os.path.join(DIRETORIO_DADOS, "sinan", f"notificacoes_exantematicas_sinan_{ano}.csv"),
                   COLUNAS_SINAN)
        print(f"  SINAN {ano}: {len(df)} notificações")

def filtro_sarampo(fatias):
    """AIHs com o CID de sarampo em qualquer diagnóstico (o cálculo decide se usa os secundários)."""
    mascara = np.zeros(len(fatias["DIAG_PRINC"]), dtype=bool)
    for col in COLUNAS_DIAG:
        if col in fatias:
            mascara |= np.char.startswith(fatias[col], CID.encode())
    return mascara

def sih_do_arquivo(uf, ano, mes):
    df = recorte_em_cache("sih_recortes_espec", f"RD{uf}{ano % 100:02d}{mes:02d}.dbc", FTP_SIH,
                          lambda caminho: ler_dbc(caminho, COLUNAS_SIH, filtro=filtro_sarampo))
    if df is not None:
        df["UF"] = uf
        df["COMPETEN"] = f"{ano}{mes:02d}"            # o nome do arquivo é a competência
    return df

def exportar_sih(anos, ufs):
    """Um CSV por ano de competência, com as AIHs de sarampo."""
    tarefas = [(uf, ano, mes) for ano in anos for uf in ufs for mes in range(1, 13)]
    resultados = processar_em_paralelo(sih_do_arquivo, tarefas, "SIH")
    for ano in anos:
        partes = [df for (_, a, _), df in resultados.items() if a == ano and df is not None]
        if not partes:
            print(f"  SIH {ano}: não publicado")
            continue
        df = pd.concat(partes, ignore_index=True).fillna("")   # arquivos antigos não têm DIAGSEC1-9
        gravar_csv(df.sort_values(["COMPETEN", "UF"]),
                   os.path.join(DIRETORIO_DADOS, "sih", f"internacoes_sarampo_sih_{ano}.csv"), COLUNAS_SIH)
        print(f"  SIH {ano}: {len(df)} AIHs com {CID}")

def filtro_com_uti(fatias):
    """AIHs com algum dia de UTI (UTI_MES_TO diferente de 0 e de vazio)."""
    dias = np.char.strip(fatias["UTI_MES_TO"])
    return (dias != b"") & (np.char.lstrip(dias, b"0") != b"")

def somar_dias_uti(caminho):
    """Dias de UTI de todas as AIHs do arquivo, somados por tipo de UTI (MARCA_UTI)."""
    df = ler_dbc(caminho, ["UTI_MES_TO", "MARCA_UTI"], filtro=filtro_com_uti)
    df["DIAS_UTI"] = pd.to_numeric(df["UTI_MES_TO"], errors="coerce").fillna(0).astype(int)
    return df.groupby("MARCA_UTI", as_index=False)["DIAS_UTI"].sum()

def uti_do_arquivo(uf, ano, mes):
    df = recorte_em_cache("sih_uti", f"RD{uf}{ano % 100:02d}{mes:02d}.dbc", FTP_SIH, somar_dias_uti)
    if df is not None:
        df["UF"] = uf
        df["COMPETEN"] = f"{ano}{mes:02d}"            # o nome do arquivo é a competência
    return df

def exportar_sih_uti(anos, ufs):
    """Um CSV por ano de competência, com os dias de UTI de todas as AIHs por UF x tipo de UTI."""
    tarefas = [(uf, ano, mes) for ano in anos for uf in ufs for mes in range(1, 13)]
    resultados = processar_em_paralelo(uti_do_arquivo, tarefas, "SIH UTI")
    for ano in anos:
        partes = [df for (_, a, _), df in resultados.items() if a == ano and df is not None]
        if not partes:
            print(f"  SIH UTI {ano}: não publicado")
            continue
        df = pd.concat(partes, ignore_index=True)
        gravar_csv(df.sort_values(["COMPETEN", "UF", "MARCA_UTI"]),
                   os.path.join(DIRETORIO_DADOS, "sih", f"dias_uti_sih_{ano}.csv"), COLUNAS_SIH_UTI)
        print(f"  SIH UTI {ano}: {df['DIAS_UTI'].astype(int).sum():,} dias de UTI")

def somar_leitos(caminho):
    quantidades = [c for c in COLUNAS_CNES if c.startswith("QT_")]
    lt = ler_dbc(caminho, ["COMPETEN", "TP_LEITO", "CODLEITO"] + quantidades)
    for col in quantidades:
        lt[col] = pd.to_numeric(lt[col], errors="coerce").fillna(0).astype(int)
    return lt.groupby(["COMPETEN", "TP_LEITO", "CODLEITO"], as_index=False)[quantidades].sum()

def cnes_do_arquivo(uf, ano, mes):
    df = recorte_em_cache("cnes_leitos", f"LT{uf}{ano % 100:02d}{mes:02d}.dbc", FTP_CNES_LT, somar_leitos)
    if df is not None:
        df.insert(0, "UF", uf)
    return df

def exportar_cnes(anos, ufs):
    """Um CSV por ano, com os leitos somados por competência x UF x tipo de leito."""
    tarefas = [(uf, ano, mes) for ano in anos for uf in ufs for mes in range(1, 13)]
    resultados = processar_em_paralelo(cnes_do_arquivo, tarefas, "CNES")
    for ano in anos:
        partes = [df for (_, a, _), df in resultados.items() if a == ano and df is not None]
        if not partes:
            print(f"  CNES {ano}: não publicado")
            continue
        df = pd.concat(partes, ignore_index=True)
        gravar_csv(df.sort_values(["COMPETEN", "UF", "TP_LEITO", "CODLEITO"]),
                   os.path.join(DIRETORIO_DADOS, "cnes", f"leitos_cnes_{ano}.csv"), COLUNAS_CNES)
        print(f"  CNES {ano}: {df['COMPETEN'].nunique()} competências")

def exportar_populacao(anos):
    """Um CSV por ano, com a população estimada por UF. Os zips trazem um DBF comum."""
    for ano in anos:
        caminho = baixar(f"POPSBR{ano % 100:02d}.zip", FTP_POP)
        if caminho is None:
            print(f"  População {ano}: não publicada")
            continue
        with zipfile.ZipFile(caminho) as z:
            pop = ler_dbf(z.read(z.namelist()[0]), ["COD_MUN", "POP"])
        pop["UF"] = pop["COD_MUN"].str[:2].map(UF_IBGE)
        pop["POP"] = pd.to_numeric(pop["POP"], errors="coerce").fillna(0).astype(int)
        por_uf = pop.groupby("UF", as_index=False)["POP"].sum()
        gravar_csv(por_uf, os.path.join(DIRETORIO_DADOS, "populacao", f"populacao_por_uf_{ano}.csv"),
                   COLUNAS_POPULACAO)
        print(f"  População {ano}: {por_uf['POP'].sum():,}")


# =============================================================================
# 3. EXECUCAO
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--anos", type=int, nargs="+", default=ANOS)
    parser.add_argument("--bases", nargs="+", default=BASES, choices=BASES)
    args = parser.parse_args()

    print(f"Destino: {DIRETORIO_DADOS}")
    if "sinan" in args.bases:
        print("SINAN...")
        exportar_sinan(args.anos)
    if "sih" in args.bases:
        print("SIH...")
        exportar_sih(args.anos, TODAS_UFS)
    if "sih_uti" in args.bases:
        print("SIH (dias de UTI)...")
        exportar_sih_uti(args.anos, TODAS_UFS)
    if "cnes" in args.bases:
        print("CNES...")
        exportar_cnes(args.anos, TODAS_UFS)
    if "populacao" in args.bases:
        print("População...")
        exportar_populacao(args.anos)


if __name__ == "__main__":
    main()
