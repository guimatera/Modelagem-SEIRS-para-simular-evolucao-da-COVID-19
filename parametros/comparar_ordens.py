"""
Compara, por R2 e RMSE, ajustes de ordem 1 a 5 das relações da Etapa A de ajustar_modelo.py.
Todas continuam lineares nos coeficientes: x = (A^T A)^-1 A^T y, sem limites.

    polinomio_x:            y = a_1 x + ... + a_k x^k
    polinomio_x_intercepto: y = a_0 + a_1 x + ... + a_k x^k
    coeficiente_no_tempo:   y = x * (a_0 + a_1 t + ... + a_(k-1) t^(k-1)), o parâmetro muda com o tempo

R2 e RMSE saem dentro da amostra e fora dela, deixando de fora um bloco contíguo por vez.
Dentro da amostra, subir a ordem nunca piora o ajuste; a comparação de ordens usa o valor fora da amostra.

Hipóteses declaradas:
    - coeficiente/ordem > 1 deixa de ser a constante do modelo: p_H, c_1 etc. passam a depender de x ou de t.
    - "coef_min" e "coef_max" são y_previsto / x nos dados: fora de [0, 1], uma fração não tem sentido.

    python parametros/comparar_ordens.py
"""

import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)

import ajustar_modelo as am

SAIDA = os.path.join(am.DADOS, "resultados", "ajustes", "comparacao_ordens.csv")
ORDENS = range(1, 6)
N_BLOCOS = 5
FAMILIAS = ("polinomio_x", "polinomio_x_intercepto", "coeficiente_no_tempo")


def matriz(familia, x, t, ordem):
    if familia == "polinomio_x":
        return np.column_stack([x ** k for k in range(1, ordem + 1)])
    if familia == "polinomio_x_intercepto":
        return np.column_stack([x ** k for k in range(ordem + 1)])
    return np.column_stack([x * t ** k for k in range(ordem)])


def resolver(A, y):
    escala = np.linalg.norm(A, axis=0)
    B = A / escala
    return np.linalg.solve(B.T @ B, B.T @ y) / escala


def metricas(y, previsto):
    erro = float(((y - previsto) ** 2).sum())
    return 1 - erro / float(((y - y.mean()) ** 2).sum()), np.sqrt(erro / y.size)


def avaliar(familia, x, y, t, ordem):
    # x e t normalizados para o condicionamento; a previsão não muda
    x_n = x / x.max()
    A = matriz(familia, x_n, t, ordem)
    if A.shape[1] >= y.size:
        return None
    a = resolver(A, y)
    previsto = A @ a
    r2, rmse = metricas(y, previsto)

    fora = np.full(y.size, np.nan)
    for bloco in np.array_split(np.arange(y.size), N_BLOCOS):
        treino = np.setdiff1d(np.arange(y.size), bloco)
        fora[bloco] = A[bloco] @ resolver(A[treino], y[treino])
    r2_fora, rmse_fora = metricas(y, fora)

    coef = previsto[x > 0] / x[x > 0]
    B = A / np.linalg.norm(A, axis=0)
    return {"r2": r2, "rmse": rmse, "r2_fora": r2_fora, "rmse_fora": rmse_fora, "coef_min": coef.min(),
            "coef_max": coef.max(), "cond_AtA": float(np.linalg.cond(B.T @ B)), "n": y.size}


def tempo(meses):
    ordinais = np.array([m.ordinal for m in meses], float)
    return (ordinais - ordinais.min()) / max(ordinais.max() - ordinais.min(), 1.0)


def main():
    series, sinan, sih, pop, _, doses, pop_faixas = am.carregar_dados()
    pi = float(series["mortes"].reindex(pd.period_range(am.T0, am.FIM, freq="M")).sum()
               / ((am.FIM - am.T0).days + 1) / pop[2018])

    sistemas = {}
    for nome, (n, y) in am.sistemas_clinicos(sinan, sih).items():
        sistemas[nome, "2007-2026"] = (n.values.astype(float), y.values.astype(float), tempo(n.index))
    fim_longo = pd.Period(am.DOSES_FIM, "M").end_time
    for janela, fim in ((f"{am.T0.year}-{am.FIM.year}", am.FIM), (f"{am.T0.year}-{am.DOSES_FIM[:4]}", fim_longo)):
        for nome, (x, y, meses) in am.sistemas_vacinais(series, doses, pop_faixas, pi, fim=fim).items():
            sistemas[nome, janela] = (x, y, tempo(meses))

    linhas = []
    for (nome, janela), (x, y, t) in sistemas.items():
        for familia in FAMILIAS:
            for ordem in ORDENS:
                try:
                    r = avaliar(familia, x, y, t, ordem)
                except np.linalg.LinAlgError:
                    r = None
                if r:
                    linhas.append({"parametro": nome, "janela": janela, "familia": familia, "ordem": ordem, **r})
    tabela = pd.DataFrame(linhas)

    # Ajuste epidêmico: já é não linear; R2 e RMSE na escala original e em log
    curvas = pd.read_csv(am.SAIDA_CURVAS, sep=";")
    for rotulo, obs, mod in (("casos B", "casos_obs", "casos_B"), ("internações B", "intern_obs", "intern_B")):
        y, m = curvas[obs].values, curvas[mod].values
        r2, rmse = metricas(y, m)
        r2_log, rmse_log = metricas(np.log1p(y), np.log1p(m))
        linhas.append({"parametro": rotulo, "janela": "2016-2020", "familia": "epidemico", "ordem": np.nan, "r2": r2,
                       "rmse": rmse, "r2_log": r2_log, "rmse_log": rmse_log, "n": y.size})
    pd.DataFrame(linhas).to_csv(SAIDA, sep=";", index=False)

    pd.set_option("display.width", 220, "display.max_columns", None, "display.max_rows", None)
    print(pd.DataFrame(linhas[-2:])[["parametro", "r2", "rmse", "r2_log", "rmse_log", "n"]].round(3).to_string(index=False))
    colunas = ["parametro", "janela", "familia", "ordem", "r2", "rmse", "r2_fora", "rmse_fora", "coef_min", "coef_max", "cond_AtA", "n"]
    print(tabela[colunas].round(4).to_string(index=False))
    print(f"\nGravado: {SAIDA}")


if __name__ == "__main__":
    main()
