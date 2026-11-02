"""
Importação de casos de sarampo (termo epsilon do modelo) a partir da tabulação do
TabNet/SINAN com país provável de infecção fora do Brasil:

    dados/sinan/casos_importados_sarampo_tabnet.csv  ->  casos por ano e mês de notificação

Os dados são mensais e não identificam a introdução de cada caso. Por isso, cada
mês com ao menos um caso conta como um evento k, com n_k = casos do mês:

    dados/resultados/importacao/importacao_eventos.csv     ->  eventos (ano, mês, n_k)
    dados/resultados/importacao/importacao_parametros.csv  ->  mu, por cenário

Número de eventos no ano: Poisson com taxa mu. Essa Poisson homogênea não
reproduz os anos sem casos observados na série: compare anos_sem_evento_obs
com anos_sem_evento_poisson.

    python parametros/estimar_importacao.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import chi2, poisson

# =============================================================================
# CONFIGURACOES
# =============================================================================

RAIZ = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_ENTRADA = os.path.join(RAIZ, "..", "dados", "sinan", "casos_importados_sarampo_tabnet.csv")
ARQUIVO_EVENTOS = os.path.join(RAIZ, "..", "dados", "resultados", "importacao", "importacao_eventos.csv")
ARQUIVO_PARAMETROS = os.path.join(RAIZ, "..", "dados", "resultados", "importacao", "importacao_parametros.csv")

MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
ANOS = list(range(2007, 2027))

# A tabulação foi disponibilizada em setembro de 2026: 2026 tem exposição de 9 meses
EXPOSICAO = {ano: 1.0 for ano in ANOS}
EXPOSICAO[2026] = 9 / 12

CENARIOS = {
    "2007-2026": ANOS,
    "sem 2018-2020": [a for a in ANOS if a not in (2018, 2019, 2020)],
    "2021-2026": [a for a in ANOS if a >= 2021],
}

N_BOOTSTRAP = 2000
SEMENTE = 2026


# =============================================================================
# LEITURA
# =============================================================================

def ler_eventos(caminho):
    """Linhas de ano da tabulação em formato longo, só com os meses com casos."""
    eventos = []
    with open(caminho, encoding="latin-1") as arquivo:
        for linha in arquivo:
            campos = [c.strip('"') for c in linha.strip().split(";")]
            if len(campos) != 14 or not campos[0].isdigit():
                continue
            ano = int(campos[0])
            casos = [0 if c == "-" else int(c) for c in campos[1:13]]
            if sum(casos) != int(campos[13]):
                raise ValueError(f"Total de {ano} não confere com a soma dos meses")
            eventos += [(ano, mes, n) for mes, n in zip(MESES, casos) if n > 0]
    return pd.DataFrame(eventos, columns=["ano", "mes", "n_k"])


# =============================================================================
# ESTIMACAO
# =============================================================================

def ic_poisson(k, T, alfa=0.05):
    """Intervalo exato da taxa de Poisson com k eventos em T anos."""
    inf = chi2.ppf(alfa / 2, 2 * k) / (2 * T) if k > 0 else 0.0
    sup = chi2.ppf(1 - alfa / 2, 2 * k + 2) / (2 * T)
    return inf, sup


def estimar_cenario(nome, anos, eventos, rng):
    contagens = np.array([(eventos["ano"] == a).sum() for a in anos])
    exposicao = np.array([EXPOSICAO[a] for a in anos])
    k, T = int(contagens.sum()), float(exposicao.sum())
    mu = k / T
    ic_inf, ic_sup = ic_poisson(k, T)

    # Incerteza de mu com superdispersão: reamostragem de anos
    idx = rng.integers(0, len(anos), (N_BOOTSTRAP, len(anos)))
    mus = contagens[idx].sum(axis=1) / exposicao[idx].sum(axis=1)

    anos_sem_evento = int((contagens == 0).sum())
    return {
        "cenario": nome,
        "eventos": k,
        "anos_expostos": round(T, 2),
        "casos": int(eventos.loc[eventos["ano"].isin(anos), "n_k"].sum()),
        "mu_eventos_ano": round(mu, 3),
        "ic95_poisson_inf": round(ic_inf, 3),
        "ic95_poisson_sup": round(ic_sup, 3),
        "ic95_bootstrap_inf": round(float(np.percentile(mus, 2.5)), 3),
        "ic95_bootstrap_sup": round(float(np.percentile(mus, 97.5)), 3),
        "indice_dispersao": round(float(contagens.var(ddof=1) / contagens.mean()), 2),
        "anos_sem_evento_obs": anos_sem_evento,
        "anos_sem_evento_poisson": round(float(poisson.pmf(0, mu * exposicao).sum()), 2),
    }


def main():
    eventos = ler_eventos(ARQUIVO_ENTRADA)
    eventos.to_csv(ARQUIVO_EVENTOS, sep=";", index=False)

    rng = np.random.default_rng(SEMENTE)
    resultado = pd.DataFrame([estimar_cenario(nome, anos, eventos, rng)
                              for nome, anos in CENARIOS.items()])
    resultado.to_csv(ARQUIVO_PARAMETROS, sep=";", index=False)

    pd.set_option("display.width", 200, "display.max_columns", None)
    print(f"Casos na tabulação: {eventos['n_k'].sum()} | eventos (meses com casos): {len(eventos)}")
    print("n_k:", sorted(eventos["n_k"].tolist()))
    print(eventos.groupby("ano")["n_k"].agg(eventos="count", casos="sum").T.to_string())
    print(resultado.T.to_string())


if __name__ == "__main__":
    main()
