"""
Estima os parâmetros da dinâmica de hospitalização do sarampo a partir dos
microdados do DATASUS já exportados em CSV (ver exportar_datasus.py):

    dados/sinan/notificacoes_exantematicas_sinan_AAAA.csv  ->  p_H, sigma_I1, gamma_I1
    dados/sih/internacoes_sarampo_sih_AAAA.csv            ->  p_C, sigma_H, gamma_H, d, sigma_C, gamma_C

Fluxos do modelo:
    I1 -> H : p_H * sigma_I1 * I1          I1 -> R : (1 - p_H) * gamma_I1 * I1
    H  -> C : p_C * theta * sigma_H * H    H  -> R : (1 - p_C) * gamma_H * H
    H  -> D : p_C * (1 - theta) * sigma_H * H (UTI lotada)
    C  -> D : d * sigma_C * C              C  -> R : (1 - d) * gamma_C * C

    python parametros/calcular_parametros_hospitalares.py
    python parametros/calcular_parametros_hospitalares.py --anos 2018 2019 2020 --ufs SP PR
"""

import os
import argparse

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURACOES
# =============================================================================

RAIZ            = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DADOS = os.path.join(RAIZ, "..", "dados")                  # CSVs de exportar_datasus.py
ARQUIVO_SAIDA   = os.path.join(DIRETORIO_DADOS, "resultados", "hospitalar", "parametros_hospitalares.csv")

# Período e UFs da estimativa (padrão: todo o período exportado, Brasil inteiro)
ANOS = list(range(2007, 2027))
TODAS_UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG",
             "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR",
             "RS", "SC", "SE", "SP", "TO"]

# CID-10 do sarampo. Complicações (ex.: pneumonia, J18) às vezes entram como
# diagnóstico principal com B05 como secundário.
CID = "B05"
USAR_DIAG_SECUNDARIO = True

# Janela plausível entre início dos sintomas e internação, em dias. Fora dela
# o registro é tratado como erro de digitação.
DIAS_ATE_INTERNACAO = (0, 30)

# Códigos do SINAN (dicionário das doenças exantemáticas)
CLASSI_SARAMPO = "1"                                  # CLASSI_FIN: 1 = sarampo
HOSPIT_SIM, HOSPIT_NAO = "1", "2"                     # ID_HOSPIT
VACINA_SIM, VACINA_NAO = "1", "2"                     # CS_VACINA
EVOLUCAO_OBITO_AGRAVO = "2"                           # EVOLUCAO

# Pós-estratificação de p_H: os casos com ID_HOSPIT em branco/ignorado recebem a
# fração internada do seu estrato. Os brancos se concentram em 2019-2020 (SP e
# PA), então a média nacional dos casos completos não os representa bem.
# Cada caso usa o nível mais detalhado cujo estrato tenha pelo menos
# MIN_INFORMADOS_ESTRATO casos com ID_HOSPIT preenchido; senão, sobe um nível.
NIVEIS_ESTRATO = [["ANO", "UF"], ["ANO"], []]
MIN_INFORMADOS_ESTRATO = 20

Z95 = 1.959964


# =============================================================================
# 1. LEITURA DOS CSVS
# =============================================================================

def ler_csvs(base, prefixo, anos, colunas):
    """
    Concatena dados/<base>/<prefixo>_AAAA.csv dos `anos` pedidos
    (os que não existirem são ignorados), só com as `colunas` presentes. Tudo é
    lido como texto: os códigos do DATASUS têm zeros à esquerda.
    """
    partes = []
    for ano in anos:
        caminho = os.path.join(DIRETORIO_DADOS, base, f"{prefixo}_{ano}.csv")
        if not os.path.exists(caminho):
            continue
        partes.append(pd.read_csv(caminho, sep=";", dtype=str, keep_default_na=False,
                                  usecols=lambda c: c in colunas))
    if not partes:
        raise FileNotFoundError(f"nenhum CSV {base}/{prefixo}_AAAA.csv para {anos}; "
                                "rode antes parametros/exportar_datasus.py")
    return pd.concat(partes, ignore_index=True).fillna("")  # arquivos antigos têm menos colunas


# =============================================================================
# 2. SINAN: CASOS CONFIRMADOS DE SARAMPO
# =============================================================================

COLUNAS_SINAN = ["CLASSI_FIN", "DT_SIN_PRI", "DT_INTERNA", "ID_HOSPIT",
                 "CS_VACINA", "EVOLUCAO", "SG_UF", "SG_UF_NOT", "UF_H"]

def carregar_sinan(anos):
    """Casos confirmados com início dos sintomas nos `anos` pedidos."""
    # O arquivo é por ano de notificação; um caso do fim do ano pode ser
    # notificado no ano seguinte, então lê também o arquivo do ano seguinte.
    sinan = ler_csvs("sinan", "notificacoes_exantematicas_sinan", sorted(set(anos) | {max(anos) + 1}), COLUNAS_SINAN)
    print(f"  {len(sinan)} notificações")
    sinan = sinan[sinan["CLASSI_FIN"] == CLASSI_SARAMPO].copy()
    for col in ["DT_SIN_PRI", "DT_INTERNA"]:
        sinan[col] = pd.to_datetime(sinan[col], format="%Y%m%d", errors="coerce")
    sinan = sinan[sinan["DT_SIN_PRI"].dt.year.isin(anos)].copy()

    # Dedução lógica: hospitalização em branco/ignorada, mas com a UF do
    # hospital preenchida, só pode ser internação.
    sem_info = ~sinan["ID_HOSPIT"].isin([HOSPIT_SIM, HOSPIT_NAO])
    com_hospital = sinan["UF_H"] != ""
    sinan.loc[sem_info & com_hospital, "ID_HOSPIT"] = HOSPIT_SIM
    print(f"  {(sem_info & com_hospital).sum()} casos com ID_HOSPIT deduzido pela UF do hospital")

    # Variáveis dos estratos (UF de residência; na falta, a de notificação)
    sinan["ANO"] = sinan["DT_SIN_PRI"].dt.year
    sinan["UF"] = sinan["SG_UF"].where(sinan["SG_UF"] != "", sinan["SG_UF_NOT"])
    return sinan


# =============================================================================
# 3. SIH: INTERNACOES COM CID DE SARAMPO
# =============================================================================

COLUNAS_DIAG = ["DIAG_PRINC", "DIAG_SECUN"] + [f"DIAGSEC{k}" for k in range(1, 10)]
COLUNAS_SIH = ["UF", "COMPETEN", "N_AIH", "IDENT", "DT_INTER", "DT_SAIDA", "DIAS_PERM",
               "UTI_MES_TO", "MARCA_UTI", "MORTE", "ESPEC", "CNES"] + COLUNAS_DIAG

def carregar_sih(anos, ufs):
    """AIHs de sarampo internadas nos `anos` pedidos, uma linha por internação."""
    # O CSV é por ano de competência (mês de processamento), que pode vir meses
    # depois da internação: lê também o ano seguinte e filtra pela data de internação.
    sih = ler_csvs("sih", "internacoes_sarampo_sih", sorted(set(anos) | {max(anos) + 1}), COLUNAS_SIH)
    for col in COLUNAS_SIH:                           # arquivos antigos não têm DIAGSEC1-9
        if col not in sih:
            sih[col] = ""
    sih = sih[sih["UF"].isin(ufs)].copy()

    # O CSV traz o CID em qualquer diagnóstico; aqui se decide se os secundários contam
    diags = ["DIAG_PRINC"] + (COLUNAS_DIAG[1:] if USAR_DIAG_SECUNDARIO else [])
    sih = sih[np.logical_or.reduce([sih[c].str.startswith(CID) for c in diags])].copy()
    print(f"  {len(sih)} AIHs com {CID}")

    for col in ["DIAS_PERM", "UTI_MES_TO", "MORTE"]:
        sih[col] = pd.to_numeric(sih[col], errors="coerce").fillna(0).astype(int)
    for col in ["DT_INTER", "DT_SAIDA"]:
        sih[col] = pd.to_datetime(sih[col], format="%Y%m%d", errors="coerce")
    sih["CID_PRINCIPAL"] = sih["DIAG_PRINC"].str.startswith(CID)

    # Internações longas geram AIHs de continuação (IDENT = 5) com o mesmo
    # número: soma os dias e junta num único registro por internação.
    internacoes = (sih.groupby(["UF", "N_AIH"], as_index=False)
                      .agg(DT_INTER=("DT_INTER", "min"),
                           DT_SAIDA=("DT_SAIDA", "max"),
                           DIAS_PERM=("DIAS_PERM", "sum"),
                           UTI_DIAS=("UTI_MES_TO", "sum"),
                           MORTE=("MORTE", "max"),
                           CID_PRINCIPAL=("CID_PRINCIPAL", "max")))
    return internacoes[internacoes["DT_INTER"].dt.year.isin(anos)]


# =============================================================================
# 4. ESTIMATIVAS
# =============================================================================

def proporcao(sucessos, total):
    """Proporção com intervalo de confiança de Wilson (95%)."""
    if total == 0:
        return np.nan, np.nan, np.nan
    p = sucessos / total
    centro = (p + Z95**2 / (2 * total)) / (1 + Z95**2 / total)
    meia = Z95 * np.sqrt(p * (1 - p) / total + Z95**2 / (4 * total**2)) / (1 + Z95**2 / total)
    return p, max(0.0, centro - meia), min(1.0, centro + meia)

def taxa(duracoes):
    """
    Taxa = 1 / duração média (1/dia). O intervalo vem do IC 95% da média,
    invertido (a média maior dá a taxa menor).
    """
    duracoes = np.asarray(duracoes, dtype=float)
    n = len(duracoes)
    if n == 0 or duracoes.mean() <= 0:
        return np.nan, np.nan, np.nan
    media = duracoes.mean()
    erro = Z95 * duracoes.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    sup = 1 / (media - erro) if media - erro > 0 else np.inf
    return 1 / media, 1 / (media + erro), sup

def media_dias(duracoes):
    """Duração média em dias, com IC 95% (aproximação normal)."""
    duracoes = np.asarray(duracoes, dtype=float)
    n = len(duracoes)
    if n == 0:
        return np.nan, np.nan, np.nan
    media = duracoes.mean()
    erro = Z95 * duracoes.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return media, max(0.0, media - erro), media + erro           # duração não é negativa

def proporcao_pos_estratificada(casos, informado, sucesso):
    """
    Proporção pós-estratificada: p = soma_g (N_g / N) * p_g, em que N_g conta
    TODOS os casos do estrato g (inclusive os sem informação) e p_g vem só dos
    informados. Equivale a imputar em cada caso sem informação a proporção do
    seu estrato (hipótese MAR: dentro do estrato, a falta é aleatória).

    Cada caso usa o primeiro nível de NIVEIS_ESTRATO cujo estrato tenha ao menos
    MIN_INFORMADOS_ESTRATO informados. Variância: soma_g (N_g/N)^2 p_g(1-p_g)/n_g,
    que ignora a pequena covariância entre estratos de níveis diferentes.

    Devolve ((p, ic_inf, ic_sup), casos por nível).
    """
    casos = casos.reset_index(drop=True)
    informado = np.asarray(informado)
    sucesso = np.asarray(sucesso) & informado
    pendente = np.ones(len(casos), dtype=bool)
    celulas, por_nivel = [], {}

    for i, nivel in enumerate(NIVEIS_ESTRATO):
        ultimo = i == len(NIVEIS_ESTRATO) - 1
        chave = (casos[nivel].astype(str).agg("|".join, axis=1) if nivel
                 else pd.Series("Brasil", index=casos.index))
        estratos = pd.DataFrame({"N": pd.Series(pendente).groupby(chave).sum(),
                                 "n": pd.Series(informado).groupby(chave).sum(),
                                 "s": pd.Series(sucesso).groupby(chave).sum()})
        aceitos = estratos[(estratos["N"] > 0) & ((estratos["n"] >= MIN_INFORMADOS_ESTRATO) | ultimo)]
        celulas.append(aceitos)

        usados = pendente & chave.isin(aceitos.index).to_numpy()
        por_nivel[" x ".join(nivel) or "Brasil"] = int(usados.sum())
        pendente &= ~usados

    celulas = pd.concat(celulas)
    celulas = celulas[celulas["n"] > 0]                   # estrato sem nenhum informado não contribui
    peso = celulas["N"] / celulas["N"].sum()
    p_g = celulas["s"] / celulas["n"]
    p = (peso * p_g).sum()
    erro = Z95 * np.sqrt((peso**2 * p_g * (1 - p_g) / celulas["n"]).sum())
    return (p, max(0.0, p - erro), min(1.0, p + erro)), por_nivel

def estimar(sinan, sih):
    linhas = []

    def registrar(parametro, estimativa, n, fonte, descricao):
        valor, inf, sup = estimativa
        linhas.append(dict(parametro=parametro, valor=valor, ic95_inf=inf, ic95_sup=sup,
                           n=n, fonte=fonte, descricao=descricao))

    # --- SINAN ---------------------------------------------------------------
    hosp = sinan["ID_HOSPIT"]
    n_sim, n_nao = (hosp == HOSPIT_SIM).sum(), (hosp == HOSPIT_NAO).sum()
    registrar("p_H", proporcao(n_sim, n_sim + n_nao), n_sim + n_nao, "SINAN",
              "fração internada, entre os casos com ID_HOSPIT preenchido (sim/não)")
    registrar("p_H_minimo", proporcao(n_sim, len(sinan)), len(sinan), "SINAN",
              "limite inferior: internados / todos os confirmados (em branco = não internado)")

    for codigo, rotulo in [(VACINA_NAO, "nao_vacinados"), (VACINA_SIM, "vacinados")]:
        grupo = hosp[sinan["CS_VACINA"] == codigo]
        s, t = (grupo == HOSPIT_SIM).sum(), grupo.isin([HOSPIT_SIM, HOSPIT_NAO]).sum()
        registrar(f"p_H_{rotulo}", proporcao(s, t), t, "SINAN",
                  f"fração internada entre os {rotulo.replace('_', ' ')} (CS_VACINA)")

    informado = hosp.isin([HOSPIT_SIM, HOSPIT_NAO])
    for filtro, sufixo in [(slice(None), ""), (sinan["CS_VACINA"] == VACINA_NAO, "_nao_vacinados")]:
        grupo = sinan[filtro]
        estimativa, por_nivel = proporcao_pos_estratificada(
            grupo, informado[filtro], (hosp == HOSPIT_SIM)[filtro])
        niveis = ", ".join(f"{nome}: {n}" for nome, n in por_nivel.items())
        registrar(f"p_H{sufixo}_estratificado", estimativa, len(grupo), "SINAN",
                  f"pós-estratificado por ano x UF (casos por nível: {niveis})")

    # Tempo em I1. Só é mensurável para quem foi internado (DT_INTERNA); o SINAN
    # não tem data de cura, então o tempo até R não é observado.
    dias = (sinan["DT_INTERNA"] - sinan["DT_SIN_PRI"]).dt.days
    dias = dias[(hosp == HOSPIT_SIM) & dias.between(*DIAS_ATE_INTERNACAO)]
    registrar("dias_I1_ate_H", media_dias(dias), len(dias), "SINAN",
              "dias entre início dos sintomas e internação (observado)")
    registrar("dias_I1_ate_R", (np.nan, np.nan, np.nan), 0, "SINAN",
              "não observável: o SINAN não registra a data de cura")

    # No diagrama, I1 sai por dois fluxos: p_H*sigma_I1*I1 e (1-p_H)*gamma_I1*I1.
    # Com saídas exponenciais concorrentes, o tempo em I1 tem a mesma distribuição
    # para os dois destinos, e só a taxa total é estimável a partir das durações.
    # O estimador de máxima verossimilhança dessa taxa usa as durações observadas
    # (as dos internados); com sigma_I1 = gamma_I1 = essa taxa, p_H é a fração internada.
    registrar("sigma_I1", taxa(dias), len(dias), "SINAN",
              "taxa I1 -> H (1/dia): 1 / dias médios até a internação")
    registrar("gamma_I1", taxa(dias), len(dias), "SINAN",
              "taxa I1 -> R (1/dia): igual a sigma_I1 (tempo até R não observado; ver dias_I1_ate_R)")

    obitos = (sinan["EVOLUCAO"] == EVOLUCAO_OBITO_AGRAVO).sum()
    registrar("obitos_sinan", (obitos, np.nan, np.nan), len(sinan), "SINAN",
              "óbitos pelo agravo entre os confirmados (validação, não entra no modelo)")

    # --- SIH -----------------------------------------------------------------
    com_uti = sih["UTI_DIAS"] > 0
    obito = sih["MORTE"] == 1

    # Tempo em H (enfermaria antes da UTI ou até a alta):
    #   sem UTI           -> DIAS_PERM
    #   com UTI e óbito   -> DIAS_PERM - UTI_DIAS, supondo que o óbito ocorreu na UTI
    #                        (o SIH não informa onde)
    #   com UTI e alta    -> não separável: os dias fora da UTI podem ser antes ou
    #                        depois dela, e o SIH não registra a ordem
    dias_H_ate_R = sih.loc[~com_uti & ~obito, "DIAS_PERM"]
    dias_H_ate_C = (sih.loc[com_uti & obito, "DIAS_PERM"] - sih.loc[com_uti & obito, "UTI_DIAS"]).clip(lower=0)
    dias_H = pd.concat([sih.loc[~com_uti, "DIAS_PERM"], dias_H_ate_C])
    registrar("dias_H_ate_R", media_dias(dias_H_ate_R), len(dias_H_ate_R), "SIH",
              "dias de enfermaria das internações sem UTI com alta (observado)")
    registrar("dias_H_ate_C", media_dias(dias_H_ate_C), len(dias_H_ate_C), "SIH",
              "dias de enfermaria antes da UTI, só nos óbitos em UTI (observado)")

    # Mesma situação de I1: H sai por p_C*sigma_H*H e (1-p_C)*gamma_H*H, e só a
    # taxa total de saída é estimável a partir das durações.
    registrar("p_C", proporcao(com_uti.sum(), len(sih)), len(sih), "SIH",
              "fração das internações com dias de UTI")
    registrar("sigma_H", taxa(dias_H), len(dias_H), "SIH",
              "taxa H -> C (1/dia): 1 / dias médios em H (sem UTI + antes da UTI nos óbitos)")
    registrar("gamma_H", taxa(dias_H), len(dias_H), "SIH",
              "taxa H -> R (1/dia): igual a sigma_H (mesma taxa total de saída de H)")

    # C sai por d*sigma_C*C (óbito) e (1-d)*gamma_C*C (alta)
    dias_C = sih.loc[com_uti, "UTI_DIAS"]
    registrar("dias_C_ate_D", media_dias(sih.loc[com_uti & obito, "UTI_DIAS"]), (com_uti & obito).sum(), "SIH",
              "dias de UTI dos óbitos (observado)")
    registrar("dias_C_ate_R", media_dias(sih.loc[com_uti & ~obito, "UTI_DIAS"]), (com_uti & ~obito).sum(), "SIH",
              "dias de UTI das altas (observado)")
    registrar("d", proporcao((com_uti & obito).sum(), com_uti.sum()), com_uti.sum(), "SIH",
              "fração de óbitos entre as internações com UTI")
    registrar("sigma_C", taxa(dias_C), len(dias_C), "SIH",
              "taxa C -> D (1/dia): 1 / dias médios de UTI (óbitos + altas)")
    registrar("gamma_C", taxa(dias_C), len(dias_C), "SIH",
              "taxa C -> R (1/dia): igual a sigma_C (mesma taxa total de saída de C)")
    registrar("d_H", proporcao(sih.loc[~com_uti, "MORTE"].sum(), (~com_uti).sum()), (~com_uti).sum(), "SIH",
              "fração de óbitos entre as internações sem UTI (se relevante, criar o fluxo H -> D)")

    return pd.DataFrame(linhas)

def resumo_por_ano(sinan, sih):
    """Contagens por ano, para ver de onde vem a informação de cada parâmetro."""
    ano_sinan = sinan["DT_SIN_PRI"].dt.year
    ano_sih = sih["DT_INTER"].dt.year
    resumo = pd.DataFrame({
        "confirmados": ano_sinan.value_counts(),
        "internados_sinan": ano_sinan[sinan["ID_HOSPIT"] == HOSPIT_SIM].value_counts(),
        "obitos_sinan": ano_sinan[sinan["EVOLUCAO"] == EVOLUCAO_OBITO_AGRAVO].value_counts(),
        "internacoes_sih": ano_sih.value_counts(),
        "internacoes_uti": ano_sih[sih["UTI_DIAS"] > 0].value_counts(),
        "obitos_sih": ano_sih[sih["MORTE"] == 1].value_counts(),
    })
    resumo.index.name = "ano"
    return resumo.fillna(0).astype(int).sort_index()


# =============================================================================
# 5. EXECUCAO
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--anos", type=int, nargs="+", default=ANOS)
    parser.add_argument("--ufs", nargs="+", default=["TODAS"],
                        help="UFs do SIH, ou TODAS")
    args = parser.parse_args()
    ufs = TODAS_UFS if args.ufs == ["TODAS"] else [u.upper() for u in args.ufs]

    print(f"Anos: {args.anos[0]}-{args.anos[-1]}" if len(args.anos) > 3 else f"Anos: {args.anos}")
    print("Lendo SINAN (Brasil)...")
    sinan = carregar_sinan(args.anos)
    print(f"  {len(sinan)} casos confirmados de sarampo")

    print(f"Lendo SIH ({'Brasil' if ufs == TODAS_UFS else ', '.join(ufs)})...")
    sih = carregar_sih(args.anos, ufs)
    print(f"  {len(sih)} internações ({sih['CID_PRINCIPAL'].sum()} com {CID} como diagnóstico principal)")

    parametros = estimar(sinan, sih)
    parametros.to_csv(ARQUIVO_SAIDA, index=False, sep=";", encoding="utf-8-sig")
    resumo = resumo_por_ano(sinan, sih)
    resumo.to_csv(os.path.join(DIRETORIO_DADOS, "resultados", "hospitalar", "parametros_hospitalares_por_ano.csv"), sep=";")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)
    print()
    print(resumo.to_string())
    print()
    print(parametros[["parametro", "valor", "ic95_inf", "ic95_sup", "n", "fonte"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nParâmetros salvos em {ARQUIVO_SAIDA}")

    preenchidos = sinan["ID_HOSPIT"].isin([HOSPIT_SIM, HOSPIT_NAO]).mean()
    if preenchidos < 0.9:
        print(f"\nAtenção: só {preenchidos:.0%} dos confirmados têm ID_HOSPIT preenchido; "
              "p_H pode estar superestimado (casos graves tendem a ser mais bem investigados). "
              "Use p_H_minimo como limite inferior.")
    n_uti = (sih["UTI_DIAS"] > 0).sum()
    if n_uti < 30:
        print(f"Atenção: só {n_uti} internações com UTI; nu_C e d ficam instáveis. "
              "Considere valores da literatura.")


if __name__ == "__main__":
    main()
