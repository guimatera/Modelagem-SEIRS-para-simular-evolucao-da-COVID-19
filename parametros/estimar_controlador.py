"""
Estimação dos ganhos do controlador PD por mínimos quadrados, como no Apêndice A de
Gonzalez, Imbiriba & Jandre (2025), Biol. Cybern. 119:11.

Malha:
    planta      SEIHRVS, entrada u(t) na força de infecção: lambda = (1 - u) * beta * I / N
    saída       C(t), pacientes em UTI
    erro        e(t) = ICU(t) - C(t)
    controlador u(t) = sat_[0,1]( -Kp * e(t) - Kd * de/dt(t) ), sem atraso explícito
                (os atrasos já estão na DDE do modelo)

Forma algébrica (análoga a A1/A4 do artigo), sem intercepto:

    [u_1]   [-e_1  -de_1]
    [ . ] = [  .      .  ] [Kp]  + xi,     minimizando ||xi||^2   (A7)
    [u_n]   [-e_n  -de_n] [Kd]

Procedimento (como a Seção 2.4.1 do artigo):
    1. Simula a malha fechada com ganhos conhecidos; a importação epsilon sorteada é a
       perturbação de cada realização (papel do d(t) do artigo).
    2. Estima Kp, Kd por LS (np.linalg.lstsq) e NNLS (scipy.optimize.nnls), com
       de/dt (a) exata, gravada pela simulação, e (b) numérica, por diferença central nos
       valores diários de e(t) (como o artigo calcula d(theta)/dt).
    3. Simula de novo com os ganhos estimados (mesma realização) e compara C(t) e u(t).

Hipóteses declaradas:
    - Só entram no ajuste as amostras não saturadas (0 < u < 1): nas saturadas a relação
      linear não vale.
    - u é constante dentro de cada passo dt = 1 dia, calculado no início do passo.
    - Os ganhos S1 e S2 são valores de teste, em unidades da capacidade inicial de UTI
      (Kp em 1/leitos, Kd em dias/leitos); não são estimativas de nada observado.
    - Cenário: parâmetros padrão da GUI do SEIHRVS.py.

    python parametros/estimar_controlador.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

RAIZ = os.path.dirname(os.path.abspath(__file__))
PROJETO = os.path.dirname(RAIZ)
sys.path[:0] = [PROJETO, RAIZ]

from SEIHRVS import (IDX_C, MU_IMPORTACAO, TAMANHOS_IMPORTACAO, condicao_inicial, faixas_de_N,
                     simular, sortear_epsilon)

# =============================================================================
# CONFIGURACOES
# =============================================================================
DIRETORIO_SAIDA = os.path.join(PROJETO, "dados", "resultados", "controle")
SAIDA_ESTIMATIVAS = os.path.join(DIRETORIO_SAIDA, "estimativas_pd.csv")
SAIDA_RESUMO = os.path.join(DIRETORIO_SAIDA, "resumo_pd.csv")
SAIDA_FIGURA = os.path.join(DIRETORIO_SAIDA, "estimativas_pd.png")

# Valores padrão da GUI do SEIHRVS.py
N = 213421037
E0, I0, H0, C0 = 1, 0, 0, 0
PARAMS = {
    "R0": 12.0, "Pi": 2e-5, "Lambda": 1000.0,
    "tau_E": 12.5, "tau_IH": 4.23, "tau_IR": 8.0, "tau_HC": 6.7, "tau_HR": 4.47,
    "tau_HD": 4.8, "tau_CR": 14.0, "tau_CD": 7.9,
    "pH": 0.2, "pC": 0.032, "d": 0.0804,
    "tau_D1": 365.0, "tau_v2_1": 90.0, "tau_v2_2": 30.0,
    "e1": 0.93, "e2": 0.571, "c1": 1.0, "c2_1": 1.0, "c2_2": 1.0,
    "v1_2": 0.002, "v1_3": 0.001, "ICU": 2.43,
}
T0, TF, DT = 0, 2 * 365, 1.0

# Ganhos de teste em unidades da capacidade inicial: Kp = a / ICU_0, Kd = b / ICU_0
GANHOS_TESTE = {"S1": (0.1, 5.0), "S2": (0.5, 20.0)}
N_REALIZACOES = 20
SEMENTE = 2025


def r2(y, y_hat):
    return 1.0 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)


def simular_malha(controle, epsilon):
    x0 = condicao_inicial(faixas_de_N(N), E0, I0, H0, C0)
    fl = {}
    x, t, rt, icu = simular(x0, T0, TF, DT, PARAMS, epsilon, fluxos=fl, controle=controle)
    return {"t": t, "C": x[IDX_C], "icu": icu, "u": fl["u"], "e": fl["erro"], "de": fl["derro"]}


def estimar(sim, derivada, metodo):
    # Último passo não tem u calculado (a simulação para antes): fica de fora
    u, e = sim["u"][:-1], sim["e"][:-1]
    if derivada == "exata":
        de = sim["de"][:-1]
    else:
        de = np.gradient(sim["icu"] - sim["C"], DT)[:-1]
    usar = (u > 0.0) & (u < 1.0)
    A = np.column_stack([-e[usar], -de[usar]])
    b = u[usar]
    if metodo == "LS":
        theta = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        theta = nnls(A, b)[0]
    ajuste = A @ theta
    return {"Kp_hat": theta[0], "Kd_hat": theta[1], "n_amostras": int(usar.sum()),
            "R2": r2(b, ajuste), "RMSE": float(np.sqrt(np.mean((b - ajuste) ** 2)))}


def main():
    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)
    nt = np.arange(T0, TF, DT).size
    rng = np.random.default_rng(SEMENTE)
    epsilons = [sortear_epsilon(MU_IMPORTACAO, TAMANHOS_IMPORTACAO, nt, DT, rng)
                for _ in range(N_REALIZACOES)]

    sem_controle = simular_malha(None, epsilons[0])
    icu0 = sem_controle["icu"][0]
    print(f"Capacidade inicial de UTI: {icu0:.0f} leitos")
    print(f"Sem controle (realização 0): pico de C = {sem_controle['C'].max():.0f} "
          f"no dia {int(sem_controle['C'].argmax())}")

    linhas = []
    exemplos = {}
    for conjunto, (a, b) in GANHOS_TESTE.items():
        Kp, Kd = a / icu0, b / icu0
        for r, eps in enumerate(epsilons):
            sim = simular_malha({"Kp": Kp, "Kd": Kd}, eps)
            if not np.any((sim["u"] > 0) & (sim["u"] < 1)):
                print(f"{conjunto}/{r}: nenhuma amostra com 0 < u < 1; a malha não atua")
                continue
            for derivada in ("exata", "numerica"):
                for metodo in ("LS", "NNLS"):
                    est = estimar(sim, derivada, metodo)
                    linha = {"conjunto": conjunto, "realizacao": r, "derivada": derivada,
                             "metodo": metodo, "Kp": Kp, "Kd": Kd, **est,
                             "erro_rel_Kp": est["Kp_hat"] / Kp - 1.0,
                             "erro_rel_Kd": est["Kd_hat"] / Kd - 1.0,
                             "pico_C": sim["C"].max(), "dia_pico_C": int(sim["C"].argmax()),
                             "u_max": sim["u"].max(), "dias_u_saturado_1": int((sim["u"] >= 1).sum())}
                    # Re-simulação com os ganhos estimados (Seção 2.4.2), mesma realização;
                    # ganhos negativos (só no LS) ficam sem re-simulação
                    if est["Kp_hat"] >= 0 and est["Kd_hat"] >= 0:
                        sim_est = simular_malha({"Kp": est["Kp_hat"], "Kd": est["Kd_hat"]}, eps)
                        linha.update(
                            pico_C_est=sim_est["C"].max(),
                            RMSE_C=float(np.sqrt(np.mean((sim_est["C"] - sim["C"]) ** 2))),
                            RMSE_u=float(np.sqrt(np.mean((sim_est["u"] - sim["u"]) ** 2))))
                        # Exemplo da figura: realização 0, LS; NNLS se o LS saiu negativo
                        if r == 0 and (metodo == "LS" or (conjunto, derivada) not in exemplos):
                            exemplos[(conjunto, derivada)] = (sim, sim_est, est, metodo)
                    linhas.append(linha)
        print(f"{conjunto}: {N_REALIZACOES} realizações simuladas")

    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA_ESTIMATIVAS, sep=";", index=False)

    resumo = (df.groupby(["conjunto", "derivada", "metodo"])
                .agg(Kp=("Kp", "first"), Kd=("Kd", "first"),
                     Kp_hat_media=("Kp_hat", "mean"), Kd_hat_media=("Kd_hat", "mean"),
                     erro_rel_Kp_medio=("erro_rel_Kp", "mean"), erro_rel_Kp_dp=("erro_rel_Kp", "std"),
                     erro_rel_Kd_medio=("erro_rel_Kd", "mean"), erro_rel_Kd_dp=("erro_rel_Kd", "std"),
                     R2_medio=("R2", "mean"), RMSE_medio=("RMSE", "mean"),
                     n_amostras_medio=("n_amostras", "mean"),
                     pico_C_medio=("pico_C", "mean"),
                     RMSE_C_medio=("RMSE_C", "mean"), RMSE_u_medio=("RMSE_u", "mean"))
                .reset_index())
    resumo.to_csv(SAIDA_RESUMO, sep=";", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.4g}".format):
        print(resumo.drop(columns=["Kp", "Kd"]))
    print(f"\nGravado: {SAIDA_ESTIMATIVAS}\nGravado: {SAIDA_RESUMO}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(GANHOS_TESTE), 3, figsize=(15, 4.5 * len(GANHOS_TESTE)), squeeze=False)
    for i, conjunto in enumerate(GANHOS_TESTE):
        sim, sim_num, est_num, met_num = exemplos[(conjunto, "numerica")]
        _, sim_ex, est_ex, met_ex = exemplos[(conjunto, "exata")]
        t = sim["t"]
        ax[i, 0].plot(t, sem_controle["C"], color="gray", label="Sem controle")
        ax[i, 0].plot(t, sim["C"], color="black", label="Ganhos verdadeiros")
        ax[i, 0].plot(t, sim_ex["C"], "--", color="tab:blue", label=f"{met_ex}, dC/dt exata")
        ax[i, 0].plot(t, sim_num["C"], ":", color="tab:red", label=f"{met_num}, dC/dt numérica")
        ax[i, 0].plot(t, sim["icu"], "--", color="darkred", lw=0.8, label="Capacidade de UTI")
        ax[i, 0].set_title(f"{conjunto}: C(t)")
        ax[i, 0].set_xlabel("Dia")
        ax[i, 0].legend(fontsize=7)
        ax[i, 1].plot(t, sim["u"], color="black", label="Ganhos verdadeiros")
        ax[i, 1].plot(t, sim_ex["u"], "--", color="tab:blue", label=f"{met_ex}, dC/dt exata")
        ax[i, 1].plot(t, sim_num["u"], ":", color="tab:red", label=f"{met_num}, dC/dt numérica")
        ax[i, 1].set_title(f"{conjunto}: u(t)")
        ax[i, 1].set_xlabel("Dia")
        ax[i, 1].legend(fontsize=7)
        usar = (sim["u"][:-1] > 0) & (sim["u"][:-1] < 1)
        de_num = np.gradient(sim["icu"] - sim["C"], DT)[:-1]
        u_aj = -est_num["Kp_hat"] * sim["e"][:-1] - est_num["Kd_hat"] * de_num
        ax[i, 2].plot(sim["u"][:-1][usar], u_aj[usar], ".", ms=3, color="tab:red")
        ax[i, 2].plot([0, 1], [0, 1], color="gray", lw=0.8)
        ax[i, 2].set_title(f"{conjunto}: u simulado × ajustado\n({met_num}, dC/dt numérica, "
                           f"R² = {est_num['R2']:.4f})")
        ax[i, 2].set_xlabel("u (simulado)")
        ax[i, 2].set_ylabel("u ajustado")
    plt.tight_layout()
    plt.savefig(SAIDA_FIGURA, dpi=150)
    print(f"Gravado: {SAIDA_FIGURA}")


if __name__ == "__main__":
    main()
