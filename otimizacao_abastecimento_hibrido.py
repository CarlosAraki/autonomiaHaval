#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Otimização de custos de abastecimento de um carro híbrido ao longo de 500.000 km.

Rendimento (km por R$): gasolina 2 km/R$ (cidade) e 2,5 km/R$ (estrada); bateria 5 km/R$
(cidade) e 3 km/R$ (estrada). Custo marginal R$/km = 1 / rendimento.

Dependências: pandas, numpy, matplotlib (opcional: seaborn para estilo).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
BASE_DIR = Path(__file__).resolve().parent

# Legibilidade: rótulos e marcas dos eixos (matplotlib)
_COR_ROTULO_EIXO = "#0f172a"
_COR_MARCA_EIXO = "#1e293b"

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
except ImportError:
    sns = None  # matplotlib sozinho já atende; seaborn é opcional

# --- Constantes das premissas ---
VIDA_UTIL_KM = 500_000
KM_VIAGEM = 20
N_VIAGENS = int(VIDA_UTIL_KM / KM_VIAGEM)  # 25.000 viagens

# Rendimento: km por real (R$) — custo marginal R$/km = 1 / (km/R$)
KM_POR_REAL_GASOLINA_CIDADE = 2.0
KM_POR_REAL_GASOLINA_ESTRADA = 2.5
KM_POR_REAL_BATERIA_CIDADE = 5.0
KM_POR_REAL_BATERIA_ESTRADA = 3.0

CUSTO_KM_GASOLINA_CIDADE = 1.0 / KM_POR_REAL_GASOLINA_CIDADE  # 0,50 R$/km
CUSTO_KM_GASOLINA_ESTRADA = 1.0 / KM_POR_REAL_GASOLINA_ESTRADA  # 0,40 R$/km
CUSTO_KM_BATERIA_CIDADE = 1.0 / KM_POR_REAL_BATERIA_CIDADE  # 0,20 R$/km
CUSTO_KM_BATERIA_ESTRADA = 1.0 / KM_POR_REAL_BATERIA_ESTRADA  # 1/3 R$/km

# Autonomia (capacidade em km) — abastecimento/recarga apenas restabelece segmentos, sem custo fixo
AUTONOMIA_GASOLINA_KM = 800.0
AUTONOMIA_ELETRICA_KM = 200.0

# Caso 1: ciclo 800 km gasolina + 200 km elétrico a cada 1.000 km (custos por km conforme cidade/estrada)
KM_GAS_POR_CICLO = 800.0
KM_ELETRICO_POR_CICLO = 200.0
KM_POR_CICLO_BASELINE = KM_GAS_POR_CICLO + KM_ELETRICO_POR_CICLO  # 1000

# Caso 2: probabilidades (complementares 95% / 5%)
# Com 20 km na bateria (1 viagem elétrica restante): 95% lembra de recarregar (plug);
# 5% esquece → esta viagem sai na gasolina.
P_ESQUECER_RECARGAR_BATERIA_COM_20KM = 0.05
# Em cada viagem na gasolina: 5% lembra de recarregar a bateria (complemento dos 95% “foco gasolina”).
P_LEMBRAR_RECARGAR_APOS_VIAGEM_GASOLINA = 0.05
# Tanque vazio: 95% lembra de encher o tanque (só restabelece autonomia; custo só por km rodado);
# 5% esquece — se houver bateria, usa elétrico nesta viagem; senão, abastece obrigatoriamente.
P_LEMBRAR_ABASTECER_TANQUE_QUANDO_VAZIO = 0.95

# Unidades de 20 km (uma viagem)
N_SEG_ELETRICO = int(AUTONOMIA_ELETRICA_KM / KM_VIAGEM)  # 10
N_SEG_GASOLINA = int(AUTONOMIA_GASOLINA_KM / KM_VIAGEM)  # 40

# Marcos de utilização acumulada (km): 5.000, 10.000, …, 500.000
KM_MARCOS_ACUM = np.arange(5_000, VIDA_UTIL_KM + 1, 5_000, dtype=np.int64)
# Número de viagens de 20 km correspondente a cada marco
VIAGENS_NOS_MARCOS = (KM_MARCOS_ACUM // KM_VIAGEM).astype(np.int64)


def _custo_caso1_por_1000_km(prop_cidade: np.ndarray) -> np.ndarray:
    """Custo de um ciclo de 1.000 km (800 gas + 200 elétrico) dada a fração de uso na cidade."""
    prop_cidade = np.asarray(prop_cidade, dtype=np.float64)
    custo_gas_800 = KM_GAS_POR_CICLO * (
        prop_cidade * CUSTO_KM_GASOLINA_CIDADE
        + (1.0 - prop_cidade) * CUSTO_KM_GASOLINA_ESTRADA
    )
    custo_el_200 = KM_ELETRICO_POR_CICLO * (
        prop_cidade * CUSTO_KM_BATERIA_CIDADE
        + (1.0 - prop_cidade) * CUSTO_KM_BATERIA_ESTRADA
    )
    return custo_gas_800 + custo_el_200


def custo_caso1_por_proporcao_cidade(prop_cidade: np.ndarray) -> np.ndarray:
    """Caso 1 — Baseline: custo total em 500.000 km (500 ciclos de 1.000 km)."""
    n_ciclos = VIDA_UTIL_KM / KM_POR_CICLO_BASELINE
    return n_ciclos * _custo_caso1_por_1000_km(prop_cidade)


def custo_caso1_acumulado_ate_km(km: np.ndarray, prop_cidade: np.ndarray) -> np.ndarray:
    """Custo acumulado Caso 1 até `km` km, na proporção `prop_cidade` cidade/estrada."""
    km = np.asarray(km, dtype=np.float64)
    prop_cidade = np.asarray(prop_cidade, dtype=np.float64)
    taxa_por_km = _custo_caso1_por_1000_km(prop_cidade) / KM_POR_CICLO_BASELINE
    return km * taxa_por_km


def custo_caso3_por_proporcao_cidade(prop_cidade: np.ndarray) -> np.ndarray:
    """
    Caso 3 — Chaveamento inteligente: elétrico só na cidade, gasolina só na estrada.
    prop_cidade: fração em [0,1] (vetor ou escalar).
    """
    prop_cidade = np.asarray(prop_cidade, dtype=np.float64)
    km_cidade = VIDA_UTIL_KM * prop_cidade
    km_estrada = VIDA_UTIL_KM * (1.0 - prop_cidade)
    return (
        km_cidade * CUSTO_KM_BATERIA_CIDADE + km_estrada * CUSTO_KM_GASOLINA_ESTRADA
    )


def custo_caso3_acumulado_ate_km(km: np.ndarray, prop_cidade: np.ndarray) -> np.ndarray:
    """
    Custo acumulado Caso 3 até `km` km, com fração `prop_cidade` do percurso na cidade.
    `km` e `prop_cidade` broadcastáveis (ex.: malha 2D).
    """
    km = np.asarray(km, dtype=np.float64)
    prop_cidade = np.asarray(prop_cidade, dtype=np.float64)
    return km * (
        prop_cidade * CUSTO_KM_BATERIA_CIDADE
        + (1.0 - prop_cidade) * CUSTO_KM_GASOLINA_ESTRADA
    )


def _acumular_custo_viagem_eletrica(
    ids: np.ndarray,
    acum_cidade: np.ndarray,
    acum_estrada: np.ndarray,
) -> None:
    """Custo da viagem de KM_VIAGEM km no modo elétrico: separa contribuição se fosse toda cidade vs toda estrada."""
    if ids.size:
        acum_cidade[ids] += KM_VIAGEM * CUSTO_KM_BATERIA_CIDADE
        acum_estrada[ids] += KM_VIAGEM * CUSTO_KM_BATERIA_ESTRADA


def _acumular_custo_viagem_gasolina(
    ids: np.ndarray,
    acum_cidade: np.ndarray,
    acum_estrada: np.ndarray,
) -> None:
    if ids.size:
        acum_cidade[ids] += KM_VIAGEM * CUSTO_KM_GASOLINA_CIDADE
        acum_estrada[ids] += KM_VIAGEM * CUSTO_KM_GASOLINA_ESTRADA


def simular_caso2_monte_carlo_vetorizado(
    n_simulacoes: int = 4096,
    seed: int = 42,
    viagens_para_snapshots: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Caso 2 — Prioridade elétrica com esquecimento (comportamental).

    Custos por km conforme rendimento (gasolina/bateria × cidade/estrada); abastecer/recarregar
    só restabelece autonomia (sem pagamento fixo).

    Modelo em passos de uma viagem (20 km):
    - Autonomia elétrica: 10 segmentos; gasolina: 40 segmentos.
    - Com 1 segmento elétrico restante: 95% lembra de recarregar (carga cheia) e viaja no elétrico;
      5% esquece → esta viagem na gasolina (se possível).
    - Em modo gasolina: consome 1 segmento; após a viagem, 5% lembra de recarregar a bateria.
    - Tanque vazio: 95% enche o tanque (autonomia); 5% esquece — se houver bateria, viagem elétrica.

    Para cada trajetória, acumula-se (acum_cidade, acum_estrada): custo se todas as viagens fossem
    na cidade vs na estrada. O custo esperado para proporção p de cidade é:
    E[custo | p] = p * acum_cidade + (1 - p) * acum_estrada (linear em p; viagens i.i.d. cidade/estrada).

    Retorna (acum_cidade, acum_estrada, snap_cidade, snap_estrada) com snapshots opcionais (n_sims × n_marcos).
    """
    rng = np.random.default_rng(seed)

    acum_cidade = np.zeros(n_simulacoes, dtype=np.float64)
    acum_estrada = np.zeros(n_simulacoes, dtype=np.float64)
    elec = np.full(n_simulacoes, N_SEG_ELETRICO, dtype=np.int32)
    gas = np.full(n_simulacoes, N_SEG_GASOLINA, dtype=np.int32)
    modo_gas = np.zeros(n_simulacoes, dtype=np.bool_)

    snap_cidade: np.ndarray | None = None
    snap_estrada: np.ndarray | None = None
    snap_idx = 0
    if viagens_para_snapshots is not None:
        viagens_para_snapshots = np.asarray(viagens_para_snapshots, dtype=np.int64)
        n_ck = viagens_para_snapshots.size
        snap_cidade = np.zeros((n_simulacoes, n_ck), dtype=np.float64)
        snap_estrada = np.zeros((n_simulacoes, n_ck), dtype=np.float64)

    for trip_num in range(1, N_VIAGENS + 1):
        # --- Ramo elétrico (fora do “período gasolina” pós-esquecimento) ---
        idx_e = np.where(~modo_gas)[0]
        if idx_e.size:
            ja_viajou_eletrico = np.zeros(n_simulacoes, dtype=np.bool_)
            id_e0 = idx_e[elec[idx_e] == 0]
            if id_e0.size:
                elec[id_e0] = N_SEG_ELETRICO - 1
                _acumular_custo_viagem_eletrica(id_e0, acum_cidade, acum_estrada)
                ja_viajou_eletrico[id_e0] = True

            idx_e2 = idx_e[~ja_viajou_eletrico[idx_e]]
            if idx_e2.size:
                e_sub = elec[idx_e2]
                maior_que_1 = e_sub > 1
                igual_1 = e_sub == 1
            else:
                maior_que_1 = np.array([], dtype=np.bool_)
                igual_1 = np.array([], dtype=np.bool_)

            id_gt1 = idx_e2[maior_que_1]
            if id_gt1.size:
                elec[id_gt1] -= 1
                _acumular_custo_viagem_eletrica(id_gt1, acum_cidade, acum_estrada)

            id_eq1 = idx_e2[igual_1]
            if id_eq1.size:
                r = rng.random(id_eq1.size)
                esqueceu_recarga = r < P_ESQUECER_RECARGAR_BATERIA_COM_20KM
                lembrou_recarga = ~esqueceu_recarga

                # Esqueceu de plugar com 20 km na bateria: esta viagem seria na gasolina.
                id_f = id_eq1[esqueceu_recarga]
                if id_f.size:
                    tem_gas = gas[id_f] > 0
                    id_com_gas = id_f[tem_gas]
                    id_sem_gas = id_f[~tem_gas]
                    # Consumo de 1 segmento de gasolina ocorre no ramo gasolina (evita dupla contagem).
                    if id_com_gas.size:
                        modo_gas[id_com_gas] = True
                    if id_sem_gas.size:
                        r_ref = rng.random(id_sem_gas.size)
                        lembra_tanque = r_ref < P_LEMBRAR_ABASTECER_TANQUE_QUANDO_VAZIO
                        id_ab = id_sem_gas[lembra_tanque]
                        id_esq_tanque = id_sem_gas[~lembra_tanque]
                        if id_ab.size:
                            gas[id_ab] = N_SEG_GASOLINA
                            modo_gas[id_ab] = True
                        if id_esq_tanque.size:
                            pode_ev = elec[id_esq_tanque] > 0
                            id_ev = id_esq_tanque[pode_ev]
                            id_obrig = id_esq_tanque[~pode_ev]
                            if id_ev.size:
                                elec[id_ev] -= 1
                                _acumular_custo_viagem_eletrica(id_ev, acum_cidade, acum_estrada)
                            if id_obrig.size:
                                gas[id_obrig] = N_SEG_GASOLINA
                                modo_gas[id_obrig] = True

                id_l = id_eq1[lembrou_recarga]
                if id_l.size:
                    elec[id_l] = N_SEG_ELETRICO - 1
                    _acumular_custo_viagem_eletrica(id_l, acum_cidade, acum_estrada)

        idx_g = np.where(modo_gas)[0]
        if idx_g.size:
            sem_gas = gas[idx_g] <= 0
            ids_sem = idx_g[sem_gas]
            if ids_sem.size:
                r_ref = rng.random(ids_sem.size)
                lembra_abastecer = r_ref < P_LEMBRAR_ABASTECER_TANQUE_QUANDO_VAZIO
                id_ab = ids_sem[lembra_abastecer]
                id_esq_t = ids_sem[~lembra_abastecer]
                if id_ab.size:
                    gas[id_ab] = N_SEG_GASOLINA
                if id_esq_t.size:
                    tem_ev = elec[id_esq_t] > 0
                    id_ev = id_esq_t[tem_ev]
                    id_obrig = id_esq_t[~tem_ev]
                    if id_ev.size:
                        elec[id_ev] -= 1
                        modo_gas[id_ev] = False
                        _acumular_custo_viagem_eletrica(id_ev, acum_cidade, acum_estrada)
                    if id_obrig.size:
                        gas[id_obrig] = N_SEG_GASOLINA

            idx_g_cons = np.where(modo_gas)[0]
            if idx_g_cons.size:
                gas[idx_g_cons] -= 1
                _acumular_custo_viagem_gasolina(idx_g_cons, acum_cidade, acum_estrada)
                r2 = rng.random(idx_g_cons.size)
                lembrou = r2 < P_LEMBRAR_RECARGAR_APOS_VIAGEM_GASOLINA
                id_lembrou = idx_g_cons[lembrou]
                if id_lembrou.size:
                    elec[id_lembrou] = N_SEG_ELETRICO
                    modo_gas[id_lembrou] = False

        if (
            snap_cidade is not None
            and snap_estrada is not None
            and snap_idx < viagens_para_snapshots.size
        ):
            if trip_num == viagens_para_snapshots[snap_idx]:
                snap_cidade[:, snap_idx] = acum_cidade
                snap_estrada[:, snap_idx] = acum_estrada
                snap_idx += 1

    return acum_cidade, acum_estrada, snap_cidade, snap_estrada


def malha_superficies_3d(
    n_simulacoes: int = 4096,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monta as malhas 2D para três superfícies 3D: % cidade × km acumulado × custo (R$).

    Retorna (X_pct, Y_km, Z1, Z2, Z3), com shapes compatíveis com ``plot_surface`` / Plotly.
    """
    props = np.arange(0.01, 1.01, 0.01)
    props = np.round(props, 2)
    eixo_x_pct = props * 100.0
    X_3d, Y_3d = np.meshgrid(eixo_x_pct, KM_MARCOS_ACUM.astype(np.float64))
    km_col = KM_MARCOS_ACUM[:, np.newaxis].astype(np.float64)
    prop_row = props[np.newaxis, :].astype(np.float64)
    Z1_3d = custo_caso1_acumulado_ate_km(km_col, prop_row)

    _ac_c, _ac_e, snap_c, snap_e = simular_caso2_monte_carlo_vetorizado(
        n_simulacoes=n_simulacoes,
        seed=seed,
        viagens_para_snapshots=VIAGENS_NOS_MARCOS,
    )
    if snap_c is None or snap_e is None:
        raise RuntimeError("Snapshots do Caso 2 não foram calculados.")

    Z2_3d = np.mean(
        snap_c[:, :, np.newaxis] * props.reshape(1, 1, -1)
        + snap_e[:, :, np.newaxis] * (1.0 - props).reshape(1, 1, -1),
        axis=0,
    )
    Z3_3d = custo_caso3_acumulado_ate_km(km_col, prop_row)
    return X_3d, Y_3d, Z1_3d, Z2_3d, Z3_3d


def formatar_reais_br(x: float, _pos: int) -> str:
    """Formatação do eixo Y em Reais (estilo brasileiro com separador de milhar)."""
    s = f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def main() -> None:
    # Proporção cidade: 1% a 100%
    props = np.arange(0.01, 1.01, 0.01)
    props = np.round(props, 2)
    eixo_x_pct = props * 100.0

    # Caso 1: baseline depende da proporção cidade/estrada (800 km gas + 200 km elétrico / 1.000 km)
    y1 = custo_caso1_por_proporcao_cidade(props)

    # Caso 3: elétrico na cidade, gasolina na estrada
    y3 = custo_caso3_por_proporcao_cidade(props)

    # Caso 2: Monte Carlo — custo esperado em p: p*acum_cidade + (1-p)*acum_estrada
    print("Executando simulação Monte Carlo do Caso 2 (vetorizada, com marcos de km)...")
    acum_c2_cidade, acum_c2_estrada, snap_c2_cidade, snap_c2_estrada = (
        simular_caso2_monte_carlo_vetorizado(
            n_simulacoes=4096,
            seed=42,
            viagens_para_snapshots=VIAGENS_NOS_MARCOS,
        )
    )
    if snap_c2_cidade is None or snap_c2_estrada is None:
        raise RuntimeError("Snapshots do Caso 2 não foram calculados.")

    props_col = props[:, np.newaxis]
    y2 = np.mean(props_col * acum_c2_cidade + (1.0 - props_col) * acum_c2_estrada, axis=1)

    # Desvio padrão amostral ao fixar 50% cidade (referência para o resumo)
    custo_ref_p50 = 0.5 * acum_c2_cidade + 0.5 * acum_c2_estrada
    dp_c2 = float(np.std(custo_ref_p50, ddof=1)) if acum_c2_cidade.size > 1 else 0.0

    df = pd.DataFrame(
        {
            "proporcao_cidade": props,
            "proporcao_cidade_pct": eixo_x_pct,
            "caso1_baseline_reais": y1,
            "caso2_esquecimento_reais": y2,
            "caso3_inteligente_reais": y3,
        }
    )

    # --- Gráfico: fill_between com transparência (comparável a area chart) ---
    if sns is None:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            try:
                plt.style.use("seaborn-whitegrid")
            except OSError:
                plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.fill_between(
        eixo_x_pct,
        y1,
        alpha=0.45,
        color="#2ecc71",
        label="Caso 1: Ciclo fixo (baseline)",
    )
    ax.plot(eixo_x_pct, y1, color="#27ae60", linewidth=2)

    ax.fill_between(
        eixo_x_pct,
        y2,
        alpha=0.45,
        color="#e74c3c",
        label="Caso 2: Elétrico + esquecimento (esperança MC)",
    )
    ax.plot(eixo_x_pct, y2, color="#c0392b", linewidth=2)

    ax.fill_between(
        eixo_x_pct,
        y3,
        alpha=0.45,
        color="#3498db",
        label="Caso 3: Chaveamento inteligente",
    )
    ax.plot(eixo_x_pct, y3, color="#2980b9", linewidth=2)

    ax.set_xlabel(
        "Proporção de Uso na Cidade (%)",
        fontsize=12,
        color=_COR_ROTULO_EIXO,
        fontweight="semibold",
    )
    ax.set_ylabel(
        "Custo Total Acumulado em 500.000 km (R$)",
        fontsize=12,
        color=_COR_ROTULO_EIXO,
        fontweight="semibold",
    )
    ax.set_title(
        "Comparação de custos de abastecimento — híbrido (500.000 km, viagens de 20 km)",
        fontsize=13,
        pad=12,
        color=_COR_ROTULO_EIXO,
        fontweight="semibold",
    )
    ax.set_xlim(eixo_x_pct.min(), eixo_x_pct.max())
    ax.tick_params(axis="both", colors=_COR_MARCA_EIXO, labelsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(formatar_reais_br))
    ax.grid(True, alpha=0.35)
    leg2d = ax.legend(loc="best", framealpha=0.92)
    for t in leg2d.get_texts():
        t.set_color(_COR_MARCA_EIXO)

    plt.tight_layout()
    out_path = BASE_DIR / "custo_abastecimento_hibrido.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura salva em: {out_path}")

    # --- Gráfico 3D: % cidade × km acumulado × custo (R$) ---
    X_3d, Y_3d = np.meshgrid(eixo_x_pct, KM_MARCOS_ACUM.astype(np.float64))

    km_col = KM_MARCOS_ACUM[:, np.newaxis].astype(np.float64)
    prop_row = props[np.newaxis, :].astype(np.float64)
    Z1_3d = custo_caso1_acumulado_ate_km(km_col, prop_row)

    Z2_3d = np.mean(
        snap_c2_cidade[:, :, np.newaxis] * props.reshape(1, 1, -1)
        + snap_c2_estrada[:, :, np.newaxis] * (1.0 - props).reshape(1, 1, -1),
        axis=0,
    )

    Z3_3d = custo_caso3_acumulado_ate_km(km_col, prop_row)

    fig3d = plt.figure(figsize=(12, 8))
    ax3d = fig3d.add_subplot(111, projection="3d")

    surf_kw = {"linewidth": 0, "antialiased": True, "alpha": 0.5}
    ax3d.plot_surface(
        X_3d, Y_3d, Z1_3d, color="#2ecc71", label="Caso 1", **surf_kw
    )
    ax3d.plot_surface(
        X_3d, Y_3d, Z2_3d, color="#e74c3c", label="Caso 2", **surf_kw
    )
    ax3d.plot_surface(
        X_3d, Y_3d, Z3_3d, color="#3498db", label="Caso 3", **surf_kw
    )

    ax3d.set_xlabel(
        "Proporção de uso na cidade (%)",
        fontsize=10,
        labelpad=8,
        color=_COR_ROTULO_EIXO,
        fontweight="semibold",
    )
    ax3d.set_ylabel(
        "Utilização acumulada (km)",
        fontsize=10,
        labelpad=8,
        color=_COR_ROTULO_EIXO,
        fontweight="semibold",
    )
    ax3d.set_zlabel(
        "Custo acumulado (R$)",
        fontsize=10,
        labelpad=10,
        color=_COR_ROTULO_EIXO,
        fontweight="semibold",
    )
    ax3d.set_title(
        "Superfícies de custo acumulado — marcos a cada 5.000 km até 500.000 km",
        fontsize=12,
        pad=16,
        color=_COR_ROTULO_EIXO,
        fontweight="semibold",
    )
    ax3d.tick_params(axis="x", colors=_COR_MARCA_EIXO, labelsize=9)
    ax3d.tick_params(axis="y", colors=_COR_MARCA_EIXO, labelsize=9)
    ax3d.tick_params(axis="z", colors=_COR_MARCA_EIXO, labelsize=9)
    ax3d.zaxis.set_major_formatter(FuncFormatter(formatar_reais_br))
    ax3d.view_init(elev=22, azim=-58)

    # Legenda manual (plot_surface não suporta legend padrão)
    leg = [
        Patch(facecolor="#2ecc71", edgecolor="#27ae60", alpha=0.6, label="Caso 1: Ciclo fixo"),
        Patch(facecolor="#e74c3c", edgecolor="#c0392b", alpha=0.6, label="Caso 2: Elétrico + esquecimento (MC)"),
        Patch(facecolor="#3498db", edgecolor="#2980b9", alpha=0.6, label="Caso 3: Chaveamento inteligente"),
    ]
    leg3d = ax3d.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    for t in leg3d.get_texts():
        t.set_color(_COR_MARCA_EIXO)

    plt.tight_layout()
    out_3d = BASE_DIR / "custo_abastecimento_hibrido_3d.png"
    plt.savefig(out_3d, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura 3D salva em: {out_3d}")

    # --- Resumo no console ---
    print("\n=== Resumo — Custo total em 500.000 km (R$) ===\n")
    for nome, y in [
        ("Caso 1 — Ciclo fixo (baseline)", y1),
        ("Caso 2 — Elétrico + esquecimento (média MC)", y2),
        ("Caso 3 — Chaveamento inteligente", y3),
    ]:
        print(f"{nome}")
        print(f"  Mínimo: R$ {y.min():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        print(f"  Máximo: R$ {y.max():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        print()

    print(
        f"Caso 2 — Desvio padrão entre simulações com 50% cidade (n=4096): "
        f"R$ {dp_c2:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )
    print("\nPrimeiras linhas do DataFrame:")
    print(df.head(3).to_string(index=False))
    print("...")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
