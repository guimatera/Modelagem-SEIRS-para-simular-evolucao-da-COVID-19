"""
Ajuste do modelo SEIHRVS para o estado de São Paulo nas ordens de ajustar_condicoes_iniciais.py
(1: R0; 2: + s; 3: + s_1-4, s_5-59; 4: uma fração por faixa; 5: + I0; 10: + s_60, E0, I0, H0, C0, f_S1),
mínimos quadrados com limites sobre os casos diários de SP.

Medido para SP (dados/sao_paulo/ e SINAN):
    - População por faixa: projeção do IBGE (edição 2024) de SP
    - Casos e internações diários: SINAN, UF de residência 35 (na falta, a de notificação)
    - tau_IH: média dos dias entre início dos sintomas e internação, nos casos de SP
    - Importados: TabNet de SP, casos confirmados com país de infecção fora do Brasil ou UF de
      infecção fora de SP, por mês de notificação; cada mês é um evento no dia 15

Hipóteses declaradas (não há dado de SP no projeto):
    - Nascimentos: série mensal do Brasil x (população de 0 ano de SP / do Brasil) no ano
    - Óbitos: mesma taxa por pessoa do Brasil (série do Brasil x população de SP / do Brasil)
    - Vacinação: c_1, v_1,g e c_2,g ajustados no Brasil (taxas por pessoa e por coorte)
    - Mês de notificação do importado tratado como o mês da importação
    - Mistura homogênea dentro do estado; demais hipóteses de ajustar_modelo.py

    python parametros/ajustar_sao_paulo.py [ordens...]
"""

import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)

import ajustar_condicoes_iniciais as aci
import ajustar_modelo as am
import calcular_parametros_hospitalares as hosp
from SEIHRVS import IDX_D_DIS

PASTA_SP = os.path.join(am.DADOS, "sao_paulo")
UF_SP = "35"
ARQUIVOS_IMPORTADOS = ("casos_importados_sarampo_sp_outros_paises_por_mes_notificacao_2007_2026.csv",
                       "casos_importados_sarampo_sp_outras_uf_por_mes_notificacao_2007_2026.csv")
# Ordem 5 contém a ordem 4 (I0 = 0): parte também da solução da ordem 4 em SP, porque as partidas
# sorteadas ficaram presas em I0 alto
aci.PARTIDAS_EXTRAS[5] = [{"R0": 13.64, "s_1_4": 0.3851, "s_5_29": 0.163, "s_30_59": 5.872e-05, "I0": 0.0}]
SAIDA = os.path.join(am.DADOS, "resultados", "ajustes", "ajuste_sao_paulo.csv")
SAIDA_CURVAS = os.path.join(am.DADOS, "resultados", "ajustes", "ajuste_sao_paulo_curvas.csv")
SAIDA_FIGURA = os.path.join(am.DADOS, "resultados", "ajustes", "ajuste_sao_paulo.png")


def ler_projecao_sp(faixa):
    return am.ler_projecao(faixa, os.path.join(PASTA_SP, f"projecao_ibge_sp_idade_{faixa}.csv"))


def populacao_por_faixa_sp():
    zero, um_a_quatro = ler_projecao_sp("0_anos"), ler_projecao_sp("1_a_4_anos")
    return {1: {a: zero[a] + um_a_quatro[a] for a in zero}, 2: ler_projecao_sp("5_a_29_anos"),
            3: ler_projecao_sp("30_a_59_anos"), 60: ler_projecao_sp("60_anos_ou_mais")}


def ler_importados_sp():
    """Eventos (ano, mês, n_k) somando os dois arquivos do TabNet de SP (ano x mês de notificação)."""
    eventos = []
    for nome in ARQUIVOS_IMPORTADOS:
        linhas = open(os.path.join(PASTA_SP, nome), encoding="latin-1").read().splitlines()
        inicio = next(i for i, linha in enumerate(linhas) if linha.startswith('"Ano'))
        fim = next(i for i in range(inicio, len(linhas)) if linhas[i].startswith('"Total'))
        tabela = pd.read_csv(io.StringIO("\n".join(linhas[inicio:fim])), sep=";", dtype=str)
        for _, linha in tabela.iterrows():
            for mes in tabela.columns:
                if mes in am.MESES and linha[mes].strip().isdigit():
                    eventos.append({"ano": int(linha.iloc[0]), "mes": mes, "n_k": float(linha[mes])})
    return pd.DataFrame(eventos).groupby(["ano", "mes"], as_index=False)["n_k"].sum()


def escalar(serie, numerador, denominador):
    """Série mensal do Brasil x razão SP/Brasil do ano."""
    serie = serie[[p.year in numerador for p in serie.index]]
    return pd.Series([v * numerador[p.year] / denominador[p.year] for p, v in serie.items()], index=serie.index)


def main():
    ordens = [int(a) for a in sys.argv[1:]] or list(aci.ORDENS)
    print("Lendo dados")
    series_br = {
        "nascimentos": am.ler_serie_mensal(os.path.join(am.DADOS, "series_mensais", "nascimentos_mensais_brasil.csv")),
        "mortes": am.ler_serie_mensal(os.path.join(am.DADOS, "series_mensais", "obitos_mensais_brasil.csv")),
    }
    faixas_br = am.populacao_por_faixa()
    pop_br = am.populacao_por_ano(faixas_br)
    faixas_sp = populacao_por_faixa_sp()
    pop_sp = am.populacao_por_ano(faixas_sp)
    zero_br, zero_sp = am.ler_projecao("0_anos"), ler_projecao_sp("0_anos")
    series_sp = {"nascimentos": escalar(series_br["nascimentos"], zero_sp, zero_br),
                 "mortes": escalar(series_br["mortes"], pop_sp, pop_br)}

    sinan = hosp.carregar_sinan(am.ANOS_HOSP)
    sinan_sp = sinan[sinan["UF"] == UF_SP].copy()
    eventos = ler_importados_sp()

    # Vacinação: taxas do Brasil (hipótese)
    pi_br = float(series_br["mortes"].reindex(pd.period_range(am.T0, am.FIM, freq="M")).sum()
                  / ((am.FIM - am.T0).days + 1) / pop_br[2018])
    vacina = am.etapa_vacinal(series_br, am.ler_doses_grupos(), faixas_br, pi_br)

    atraso_br = am.etapa_atraso_internacao(sinan)
    atraso = am.etapa_atraso_internacao(sinan_sp)
    tau_IH = float(atraso["x"][0])
    am.imprimir_linear("tau_IH SP (dias)", atraso, ["tau_IH"])
    am.imprimir_linear("tau_IH Brasil (dias), comparação", atraso_br, ["tau_IH"])

    n_0, n_1_4 = zero_sp[am.T0.year], ler_projecao_sp("1_a_4_anos")[am.T0.year]
    s_cobertura = am.s0_por_cobertura(am.ler_cobertura())
    problema = aci.ProblemaCI(series_sp, sinan_sp, pop_sp, eventos, vacina, faixas_sp, tau_IH)
    prob_t = aci.ProblemaCI(series_sp, sinan_sp, pop_sp, eventos, vacina, faixas_sp, tau_IH, fim=am.FIM_TREINO)
    for p in (problema, prob_t):
        p.configurar(n_0, n_1_4, s_cobertura)

    n = problema.n_dias
    print(f"\nSP em {am.T0.date()}: N={problema.N:.0f} faixas={np.round(problema.faixas).astype(int)}")
    print(f"Lambda={problema.base['Lambda']:.1f}/dia Pi={problema.base['Pi']:.3e}/dia")
    print(f"Dias={n}, casos={problema.y_c[:n].sum():.0f}, internações={problema.y_h[:n].sum():.0f}, "
          f"importados={problema.importados.sum():.0f} em {int((problema.importados > 0).sum())} eventos")
    print(f"Janela {am.T0.date()} a {am.FIM.date()}, treino até {am.FIM_TREINO.date()} "
          f"(casos no treino={prob_t.y_c[:prob_t.n_dias].sum():.0f}); ordens {ordens}")

    with ProcessPoolExecutor() as executor:
        linhas = list(executor.map(aci.rodar, [(o, problema, prob_t) for o in ordens]))

    datas = problema.dias[:n]
    curvas = pd.DataFrame({"dia": datas, "casos_obs": problema.y_c[:n], "intern_obs": problema.y_h[:n],
                           "importados": problema.importados[:n]})
    na_janela = sinan_sp["DT_SIN_PRI"].between(am.T0, am.FIM)
    obitos_sinan = int((sinan_sp.loc[na_janela, "EVOLUCAO"] == hosp.EVOLUCAO_OBITO_AGRAVO).sum())
    for linha in linhas:
        ordem = linha["ordem"]
        curvas[f"casos_ordem_{ordem}"] = linha.pop("curva")
        sim = problema.simular(problema.completar({k: linha[k] for k in aci.ORDENS[ordem]}))
        curvas[f"intern_ordem_{ordem}"] = sim["h"][:n]
        linha.update(tau_IH=tau_IH, N=problema.N, Lambda=problema.base["Lambda"], Pi=problema.base["Pi"],
                     importados=float(problema.importados.sum()), obitos_modelo=float(sim["x"][IDX_D_DIS, -1]),
                     obitos_sinan=obitos_sinan, intern_modelo=float(sim["h"][:n].sum()),
                     intern_obs=float(problema.y_h[:n].sum()), dia_pico_obs=int(problema.y_c[:n].argmax()))

    # Junta com as ordens já gravadas que não foram refeitas agora
    tabela = pd.DataFrame(linhas)
    if os.path.exists(SAIDA):
        antigas = pd.read_csv(SAIDA, sep=";")
        if "ordem" in antigas:
            tabela = pd.concat([antigas[~antigas["ordem"].isin(ordens)], tabela]).sort_values("ordem")
            antigas_c = pd.read_csv(SAIDA_CURVAS, sep=";")
            manter = [c for c in antigas_c.columns if c.split("_")[-1].isdigit() and int(c.split("_")[-1]) not in ordens]
            curvas = curvas.join(antigas_c[manter])
    tabela.to_csv(SAIDA, sep=";", index=False)
    curvas.to_csv(SAIDA_CURVAS, sep=";", index=False)

    for _, l in tabela[tabela["ordem"].isin(ordens)].iterrows():
        print(f"\nOrdem {l['ordem']}: " + ", ".join(f"{k}={l[k]:.4g}" for k in aci.ORDENS[l['ordem']]))
        print(f"  S por grupo={l['s_grupos']} Rt inicial={l['Rt_inicial']:.3f} no limite={l['no_limite']} "
              f"status={l['status']} avaliações={l['avaliacoes']} custos={l['custos_partidas']}")
        print(f"  R2={l['r2']:.4f} RMSE={l['rmse']:.2f} R2 log={l['r2_log']:.4f} SSE log={l['sse_log']:.1f} "
              f"AIC={l['aic']:.1f} casos={l['casos_modelo']:.0f}/{l['casos_obs']:.0f} "
              f"pico dia={l['dia_pico_modelo']}/{l['dia_pico_obs']}")
        print(f"  internações: R2={l['r2_intern']:.4f} modelo={l['intern_modelo']:.0f}/{l['intern_obs']:.0f}; "
              f"óbitos: modelo={l['obitos_modelo']:.1f}/{l['obitos_sinan']}")
        print(f"  fora da amostra: R0={l['treino_R0']:.2f} R2={l['fora_r2']:.4f} RMSE={l['fora_rmse']:.2f} "
              f"casos={l['fora_casos_modelo']:.0f}/{l['fora_casos_obs']:.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(datas, curvas["casos_obs"], ".", color="black", ms=3, label="SINAN, SP")
    ax[0].vlines(datas[problema.importados[:n] > 0], 0, problema.y_c[:n].max(), color="tab:red", lw=0.6,
                 alpha=0.5, label="Importados (dia 15)")
    ax[1].plot(datas, curvas["intern_obs"], ".", color="black", ms=3, label="SINAN, SP")
    for ordem in sorted(int(c.split("_")[-1]) for c in curvas.columns if c.startswith("casos_ordem_")):
        ax[0].plot(datas, curvas[f"casos_ordem_{ordem}"], label=f"Ordem {ordem}")
        ax[1].plot(datas, curvas[f"intern_ordem_{ordem}"], label=f"Ordem {ordem}")
    ax[0].axvline(am.FIM_TREINO, color="gray", ls=":")
    ax[0].set_ylim(0, problema.y_c[:n].max() * 1.1)
    ax[1].set_ylim(0, problema.y_h[:n].max() * 1.1)
    ax[0].set_ylabel("Casos por dia")
    ax[1].set_ylabel("Internações por dia")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(SAIDA_FIGURA, dpi=150)
    print(f"\nGravados: {SAIDA}, {SAIDA_CURVAS}, {SAIDA_FIGURA}")


if __name__ == "__main__":
    main()
