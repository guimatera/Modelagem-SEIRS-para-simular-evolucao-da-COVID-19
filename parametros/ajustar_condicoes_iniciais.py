"""
Ajuste das condições iniciais em T0 por mínimos quadrados com limites, em ordens crescentes
(número de variáveis), sobre os casos diários do SINAN de ajustar_modelo.py.

    ordem 1:  R0                              S por cobertura de D1 e D2 (comparação)
    ordem 2:  R0, s                           mesma fração suscetível em 1-59 anos
    ordem 3:  R0, s_1-4, s_5-29+30-59         duas frações
    ordem 4:  R0, s_1-4, s_5-29, s_30-59      uma fração por faixa
    ordem 5:  ordem 4 + I0                    infecciosos em T0 (no grupo 1)
    ordem 10: ordem 4 + s_60, E0, I0, H0, C0 e f_S1 (parte dos suscetíveis em S1, falha da D1)

Em todas: menores de 1 ano 100% suscetíveis, R0 nos limites do texto (12-18) e o restante dos
parâmetros como em ajustar_modelo.py. Dentro da amostra, mais variáveis nunca pioram o ajuste:
as ordens são comparadas pelo ajuste fora da amostra (treino até FIM_TREINO) e pelo AIC.

Hipóteses declaradas:
    - As frações suscetíveis ajustadas não são medidas; são as que melhor reproduzem os casos com
      este modelo (mistura homogênea entre faixas).
    - O modelo não tem compartimentos de 60+ fora de R: os suscetíveis de 60+ (ordem 10) entram em
      S e S1 do grupo 3 e passam a receber a vacinação desse grupo.
    - E0, I0, H0 e C0 entram no grupo 1, como em condicao_inicial de SEIHRVS.py.

Com --sem-surto, os dias de SURTOS ficam fora dos resíduos (padrão: ordem 3); as métricas saem
separadas para os dias do ajuste, os dias de surto (não usados) e a janela inteira. Com
--com-importados, os dias em que o modelo recebe casos importados voltam aos resíduos, mesmo
dentro dos surtos (o SINAN não registra a importação no dia 15: é a hipótese da Etapa B).

    python parametros/ajustar_condicoes_iniciais.py [ordens...] [--sem-surto] [--com-importados]
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)

import ajustar_modelo as am
from SEIHRVS import IDX_R, IDX_S, IDX_S1, IDX_V, simular

SAIDA = os.path.join(am.DADOS, "resultados", "ajustes", "ajuste_condicoes_iniciais.csv")
SAIDA_CURVAS = os.path.join(am.DADOS, "resultados", "ajustes", "ajuste_condicoes_iniciais_curvas.csv")

FRACOES = ("s", "s_1_4", "s_5_59", "s_5_29", "s_30_59", "s_60", "f_S1")
SEMENTES = ("E0", "I0", "H0", "C0")
am.LIMITES.update({**{k: (0.0, 1.0) for k in FRACOES}, **{k: (0.0, 1e4) for k in SEMENTES}})
am.USA_LOG.update({k: False for k in FRACOES + SEMENTES})

ORDENS = {1: ["R0"], 2: ["R0", "s"], 3: ["R0", "s_1_4", "s_5_59"], 4: ["R0", "s_1_4", "s_5_29", "s_30_59"],
          5: ["R0", "s_1_4", "s_5_29", "s_30_59", "I0"],
          10: ["R0", "s_1_4", "s_5_29", "s_30_59", "s_60", "E0", "I0", "H0", "C0", "f_S1"]}
# Ordens altas: mais avaliações e partidas nas melhores soluções das ordens menores
MAX_NFEV = {10: 150}
PARTIDAS_EXTRAS = {10: [{"R0": 14.83, "s_1_4": 0.2523, "s_5_29": 0.0648, "s_30_59": 0.0648},
                        {"R0": 18.0, "s_1_4": 0.761, "s_5_29": 0.0, "s_30_59": 0.0},
                        {"R0": 18.0, "s_1_4": 0.242, "s_5_29": 0.0494, "s_30_59": 0.0494}]}
N_PARTIDAS_SORTEADAS = {10: 3}

# Surtos fora dos resíduos (--sem-surto): meses com centenas a milhares de casos no SINAN
SURTOS = [("2018-02-01", "2018-10-31"), ("2019-06-01", "2020-03-31")]
SAIDA_SEM_SURTO = os.path.join(am.DADOS, "resultados", "ajustes", "ajuste_sem_surto.csv")
SAIDA_SEM_SURTO_CURVAS = os.path.join(am.DADOS, "resultados", "ajustes", "ajuste_sem_surto_curvas.csv")
SAIDA_COM_IMPORTADOS = os.path.join(am.DADOS, "resultados", "ajustes", "ajuste_sem_surto_com_importados.csv")
SAIDA_COM_IMPORTADOS_CURVAS = os.path.join(am.DADOS, "resultados", "ajustes", "ajuste_sem_surto_com_importados_curvas.csv")


class ProblemaCI(am.Problema):
    """Problema com as frações suscetíveis por faixa e os compartimentos em T0 como variáveis."""

    def configurar(self, n_0, n_1_4, s_cobertura):
        self.n_0, self.n_1_4, self.s_cobertura = n_0, n_1_4, s_cobertura

    def completar(self, teta):
        if "s0" in teta:
            return teta
        s_1_4 = teta.get("s_1_4", teta.get("s", self.s_cobertura))
        s_5_29 = teta.get("s_5_29", teta.get("s_5_59", teta.get("s", self.s_cobertura)))
        s_30_59 = teta.get("s_30_59", teta.get("s_5_59", teta.get("s", self.s_cobertura)))
        grupo_1 = (self.n_0 + self.n_1_4 * s_1_4) / (self.n_0 + self.n_1_4)
        return {**teta, "s0": (grupo_1, s_5_29, s_30_59)}

    def simular(self, teta):
        teta = self.completar(teta)
        params = dict(self.base)
        params["R0"] = float(teta["R0"])
        f_s1, s_60 = float(teta.get("f_S1", 0.0)), float(teta.get("s_60", 0.0))

        x0 = am.condicao_inicial(self.faixas, *(float(teta.get(k, 0.0)) for k in SEMENTES))
        for s, s1, v, frac in zip(IDX_S, IDX_S1, IDX_V, teta["s0"]):
            suscetiveis = frac * x0[s]
            x0[v] = x0[s] - suscetiveis
            x0[s], x0[s1] = (1.0 - f_s1) * suscetiveis, f_s1 * suscetiveis
        idosos = s_60 * x0[IDX_R]
        x0[IDX_R] -= idosos
        x0[IDX_S[2]] += (1.0 - f_s1) * idosos
        x0[IDX_S1[2]] += f_s1 * idosos

        fluxos = {}
        x, _, rt, icu = simular(x0, 0.0, float(self.dias.size), 1.0, params, self.importados, fluxos)
        return {"c": fluxos["ent_I"], "h": fluxos["ent_H"], "u": fluxos["ent_C"], "x": x, "icu": icu, "rt": rt}


class AjusteMascara(am.Ajuste):
    """Resíduos só nos dias da máscara."""

    def __init__(self, problema, nomes, mascara):
        super().__init__(problema, nomes)
        self.mascara = mascara

    def residuos(self, z):
        return super().residuos(z)[self.mascara]


def mascara_sem_surto(problema, com_importados=False):
    dias = problema.dias[:problema.n_dias]
    fora = np.ones(dias.size, bool)
    for ini, fim in SURTOS:
        fora &= ~((dias >= pd.Timestamp(ini)) & (dias <= pd.Timestamp(fim)))
    if com_importados:
        # Dias em que o modelo recebe os importados (dia 15 de cada mês com casos)
        fora |= problema.importados[:problema.n_dias] > 0
    return fora


def rodar_sem_surto(problema, ordem, com_importados=False):
    nomes = ORDENS[ordem]
    mascara = mascara_sem_surto(problema, com_importados)
    saida, saida_curvas = ((SAIDA_COM_IMPORTADOS, SAIDA_COM_IMPORTADOS_CURVAS) if com_importados
                           else (SAIDA_SEM_SURTO, SAIDA_SEM_SURTO_CURVAS))
    aj = AjusteMascara(problema, nomes, mascara)
    aj.max_nfev = MAX_NFEV.get(ordem, aj.max_nfev)
    sols = aj.otimizar(aj.partidas(am.N_PARTIDAS))
    teta = problema.completar(aj.teta(sols[0].x))
    sim = problema.simular(teta)
    n = problema.n_dias
    y, m = problema.y_c[:n], sim["c"][:n]
    linha = {"ordem": ordem, "variaveis": " ".join(nomes), "com_importados": com_importados,
             **{k: teta.get(k) for k in nomes},
             "s_grupos": tuple(round(float(s), 4) for s in teta["s0"]), "Rt_inicial": float(sim["rt"][0]),
             "dias_ajuste": int(mascara.sum()), "dias_surto": int((~mascara).sum())}
    for rotulo, sel in (("ajuste", mascara), ("surto", ~mascara), ("janela", np.ones(n, bool))):
        r2, rmse, r2_log, rmse_log = metricas(y[sel], m[sel])
        linha.update({f"{rotulo}_r2": r2, f"{rotulo}_rmse": rmse, f"{rotulo}_r2_log": r2_log,
                      f"{rotulo}_rmse_log": rmse_log, f"{rotulo}_casos_modelo": float(m[sel].sum()),
                      f"{rotulo}_casos_obs": float(y[sel].sum())})
    sse = float(((np.log1p(y[mascara]) - np.log1p(m[mascara])) ** 2).sum())
    linha.update(aic=mascara.sum() * np.log(sse / mascara.sum()) + 2 * len(nomes),
                 custos_partidas=tuple(round(2 * float(s.cost), 1) for s in sols),
                 no_limite=[k for k, z in zip(nomes, sols[0].x) if z <= 1e-3 or z >= 1 - 1e-3],
                 status=sols[0].status)
    pd.DataFrame([linha]).to_csv(saida, sep=";", index=False)
    pd.DataFrame({"dia": problema.dias[:n], "casos_obs": y, "casos_modelo": m, "no_ajuste": mascara}).to_csv(
        saida_curvas, sep=";", index=False)

    extra = ", com os dias de importação" if com_importados else ""
    print(f"\nOrdem {ordem} sem os surtos {SURTOS}{extra}: " + ", ".join(f"{k}={teta[k]:.4g}" for k in nomes))
    print(f"  S por grupo={linha['s_grupos']} Rt inicial={linha['Rt_inicial']:.3f} no limite={linha['no_limite']} "
          f"status={linha['status']} custos={linha['custos_partidas']}")
    for rotulo in ("ajuste", "surto", "janela"):
        print(f"  {rotulo}: R2={linha[rotulo + '_r2']:.4f} RMSE={linha[rotulo + '_rmse']:.3f} "
              f"R2 log={linha[rotulo + '_r2_log']:.4f} casos={linha[rotulo + '_casos_modelo']:.0f}/"
              f"{linha[rotulo + '_casos_obs']:.0f}")
    print(f"  AIC (dias do ajuste)={linha['aic']:.1f}\nGravados: {saida}, {saida_curvas}")


def metricas(y, m):
    r2, rmse = am.r2_rmse(y, m)
    r2_log, rmse_log = am.r2_rmse(np.log1p(y), np.log1p(m))
    return r2, rmse, r2_log, rmse_log


def otimizar(p, ordem):
    nomes = ORDENS[ordem]
    aj = am.Ajuste(p, nomes)
    aj.max_nfev = MAX_NFEV.get(ordem, aj.max_nfev)
    partidas = aj.partidas(N_PARTIDAS_SORTEADAS.get(ordem, am.N_PARTIDAS))
    extras = [aj.z_de({k: e.get(k, 0.0) for k in nomes}) for e in PARTIDAS_EXTRAS.get(ordem, [])]
    if extras:
        partidas = np.r_[np.array(extras), partidas]
    sols = aj.otimizar(partidas)
    return aj, sols


def rodar(args):
    ordem, problema, prob_t = args
    nomes = ORDENS[ordem]
    linha = {"ordem": ordem, "variaveis": " ".join(nomes)}

    aj_t, sols_t = otimizar(prob_t, ordem)
    teta_t = prob_t.completar(aj_t.teta(sols_t[0].x))
    sim = problema.simular(teta_t)["c"]
    n_t, n = prob_t.n_dias, problema.n_dias
    r2, rmse, r2_log, _ = metricas(problema.y_c[n_t:n], sim[n_t:n])
    linha.update(treino_R0=teta_t["R0"], fora_r2=r2, fora_rmse=rmse, fora_r2_log=r2_log,
                 fora_casos_modelo=float(sim[n_t:n].sum()), fora_casos_obs=float(problema.y_c[n_t:n].sum()))

    aj, sols = otimizar(problema, ordem)
    teta = problema.completar(aj.teta(sols[0].x))
    sim = problema.simular(teta)
    res = am.resumo(problema, teta)
    sse = float(((np.log1p(problema.y_c[:n]) - np.log1p(sim["c"][:n])) ** 2).sum())
    linha.update({k: teta.get(k) for k in nomes}, s_grupos=tuple(round(float(s), 4) for s in teta["s0"]),
                 Rt_inicial=float(sim["rt"][0]), r2=res["r2"], rmse=res["rmse"], r2_log=res["r2_log"],
                 rmse_log=res["rmse_log"], r2_intern=res["r2_intern"], sse_log=sse,
                 aic=n * np.log(sse / n) + 2 * len(nomes), casos_modelo=res["casos_modelo"], casos_obs=res["casos_obs"],
                 dia_pico_modelo=res["dia_pico_modelo"], custos_partidas=tuple(round(2 * float(s.cost), 1) for s in sols),
                 no_limite=[k for k, z in zip(nomes, sols[0].x) if z <= 1e-3 or z >= 1 - 1e-3],
                 status=sols[0].status, avaliacoes=int(sols[0].nfev), curva=sim["c"][:n])
    return linha


def main():
    com_importados = "--com-importados" in sys.argv
    sem_surto = "--sem-surto" in sys.argv or com_importados
    ordens = [int(a) for a in sys.argv[1:] if not a.startswith("--")] or ([3] if sem_surto else list(ORDENS))
    series, sinan, sih, pop, eventos, doses, pop_faixas = am.carregar_dados()
    pi = float(series["mortes"].reindex(pd.period_range(am.T0, am.FIM, freq="M")).sum()
               / ((am.FIM - am.T0).days + 1) / pop[2018])
    vacina = am.etapa_vacinal(series, doses, pop_faixas, pi)
    tau_IH = float(am.etapa_atraso_internacao(sinan)["x"][0])
    s_cobertura = am.s0_por_cobertura(am.ler_cobertura())
    n_0, n_1_4 = am.ler_projecao("0_anos")[am.T0.year], am.ler_projecao("1_a_4_anos")[am.T0.year]

    problema = ProblemaCI(series, sinan, pop, eventos, vacina, pop_faixas, tau_IH)
    prob_t = ProblemaCI(series, sinan, pop, eventos, vacina, pop_faixas, tau_IH, fim=am.FIM_TREINO)
    for p in (problema, prob_t):
        p.configurar(n_0, n_1_4, s_cobertura)
    print(f"Janela {am.T0.date()} a {am.FIM.date()}, treino até {am.FIM_TREINO.date()}; "
          f"s por cobertura={s_cobertura:.4f}; ordens {ordens}")

    if sem_surto:
        for ordem in ordens:
            rodar_sem_surto(problema, ordem, com_importados)
        return

    with ProcessPoolExecutor() as executor:
        linhas = list(executor.map(rodar, [(o, problema, prob_t) for o in ordens]))

    # Junta com as ordens já gravadas que não foram refeitas agora
    novas_curvas = {f"ordem_{l['ordem']}": l.pop("curva") for l in linhas}
    tabela = pd.DataFrame(linhas)
    curvas = pd.DataFrame({"dia": am.T0 + pd.to_timedelta(np.arange(problema.n_dias), unit="D"),
                           "casos_obs": problema.y_c[:problema.n_dias]})
    if os.path.exists(SAIDA):
        antigas = pd.read_csv(SAIDA, sep=";")
        tabela = pd.concat([antigas[~antigas["ordem"].isin(ordens)], tabela]).sort_values("ordem")
        curvas_antigas = pd.read_csv(SAIDA_CURVAS, sep=";")
        curvas = curvas.join(curvas_antigas.drop(columns=["dia", "casos_obs", *novas_curvas], errors="ignore"))
    curvas = curvas.assign(**novas_curvas)
    tabela.to_csv(SAIDA, sep=";", index=False)
    curvas.to_csv(SAIDA_CURVAS, sep=";", index=False)

    pd.set_option("display.width", 250, "display.max_columns", None)
    novas = tabela[tabela["ordem"].isin(ordens)]
    for _, l in novas.iterrows():
        print(f"\nOrdem {l['ordem']}: " + ", ".join(f"{k}={l[k]:.4g}" for k in ORDENS[l['ordem']]))
        print(f"  S por grupo={l['s_grupos']} Rt inicial={l['Rt_inicial']:.3f} R2={l['r2']:.4f} RMSE={l['rmse']:.2f} "
              f"R2 log={l['r2_log']:.4f} AIC={l['aic']:.1f} casos={l['casos_modelo']:.0f}/{l['casos_obs']:.0f}")
        print(f"  no limite={l['no_limite']} status={l['status']} avaliações={l['avaliacoes']} custos={l['custos_partidas']}")
        print(f"  fora da amostra: R0={l['treino_R0']:.2f} R2={l['fora_r2']:.4f} RMSE={l['fora_rmse']:.2f} "
              f"casos={l['fora_casos_modelo']:.0f}/{l['fora_casos_obs']:.0f}")
    print(f"\nGravados: {SAIDA}, {SAIDA_CURVAS}")


if __name__ == "__main__":
    main()
