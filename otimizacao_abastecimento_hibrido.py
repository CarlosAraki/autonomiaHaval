#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Otimização de custos de abastecimento de um carro híbrido ao longo de 500.000 km (viagens de 25 km).

Rendimento (km por R$): gasolina 2 km/R$ (cidade) e 2,5 km/R$ (estrada); bateria 5 km/R$
(cidade) e 3 km/R$ (estrada). Custo marginal R$/km = 1 / rendimento.

Caso 2 — autonomia máxima (km): gasolina 600 (cidade) / 750 (estrada); elétrico 250 (cidade) / 150 (estrada).

Dependências: pandas, numpy, matplotlib (opcional: seaborn para estilo).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
KM_VIAGEM = 25
N_VIAGENS = int(VIDA_UTIL_KM / KM_VIAGEM)  # 20.000 viagens

# Rendimento: km por real (R$) — custo marginal R$/km = 1 / (km/R$)
KM_POR_REAL_GASOLINA_CIDADE = 2.0
KM_POR_REAL_GASOLINA_ESTRADA = 2.5
KM_POR_REAL_BATERIA_CIDADE = 5.0
KM_POR_REAL_BATERIA_ESTRADA = 3.0

CUSTO_KM_GASOLINA_CIDADE = 1.0 / KM_POR_REAL_GASOLINA_CIDADE  # 0,50 R$/km
CUSTO_KM_GASOLINA_ESTRADA = 1.0 / KM_POR_REAL_GASOLINA_ESTRADA  # 0,40 R$/km
CUSTO_KM_BATERIA_CIDADE = 1.0 / KM_POR_REAL_BATERIA_CIDADE  # 0,20 R$/km
CUSTO_KM_BATERIA_ESTRADA = 1.0 / KM_POR_REAL_BATERIA_ESTRADA  # 1/3 R$/km

# Autonomia máxima (km) com tanque/bateria cheios — depende do tipo de viagem (cidade vs estrada)
AUTONOMIA_GASOLINA_ESTRADA_KM = 750.0
AUTONOMIA_GASOLINA_CIDADE_KM = 600.0
AUTONOMIA_BATERIA_ESTRADA_KM = 150.0
AUTONOMIA_BATERIA_CIDADE_KM = 250.0

# Caso 1: ciclo 800 km gasolina + 200 km elétrico a cada 1.000 km (custos por km conforme cidade/estrada)
# (padrão de distância por modo; não confundir com autonomia máxima do Caso 2)
KM_GAS_POR_CICLO = 800.0
KM_ELETRICO_POR_CICLO = 200.0
KM_POR_CICLO_BASELINE = KM_GAS_POR_CICLO + KM_ELETRICO_POR_CICLO  # 1000

# Caso 2: probabilidades (complementares 95% / 5%)
# Com autonomia elétrica só para mais uma viagem (KM_VIAGEM): 95% lembra de recarregar (plug);
# 5% esquece → esta viagem sai na gasolina.
P_ESQUECER_RECARGAR_BATERIA_ULTIMA_VIAGEM = 0.05
# Em cada viagem na gasolina: 5% lembra de recarregar a bateria (complemento dos 95% “foco gasolina”).
P_LEMBRAR_RECARGAR_APOS_VIAGEM_GASOLINA = 0.05
# Tanque vazio: 95% lembra de encher o tanque (só restabelece autonomia; custo só por km rodado);
# 5% esquece — se houver bateria, usa elétrico nesta viagem; senão, abastece obrigatoriamente.
P_LEMBRAR_ABASTECER_TANQUE_QUANDO_VAZIO = 0.95

# Marcos de utilização acumulada (km): 5.000, 10.000, …, 500.000
KM_MARCOS_ACUM = np.arange(5_000, VIDA_UTIL_KM + 1, 5_000, dtype=np.int64)
# Número de viagens de KM_VIAGEM correspondente a cada marco
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


def _auton_bat_vec(is_cidade: np.ndarray) -> np.ndarray:
    return np.where(
        is_cidade,
        AUTONOMIA_BATERIA_CIDADE_KM,
        AUTONOMIA_BATERIA_ESTRADA_KM,
    ).astype(np.float64)


def _auton_gas_vec(is_cidade: np.ndarray) -> np.ndarray:
    return np.where(
        is_cidade,
        AUTONOMIA_GASOLINA_CIDADE_KM,
        AUTONOMIA_GASOLINA_ESTRADA_KM,
    ).astype(np.float64)


def simular_caso2_monte_carlo_vetorizado(
    n_simulacoes: int = 4096,
    seed: int = 42,
    viagens_para_snapshots: np.ndarray | None = None,
    prop_cidade: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Caso 2 — Prioridade elétrica com esquecimento (comportamental).

    Autonomia depende do tipo de viagem (cada viagem é cidade ou estrada com prob. ``prop_cidade``):
    gasolina: 600 km (cidade) / 750 km (estrada) com tanque cheio; bateria: 250 km / 150 km.

    ``elec_e`` e ``gas_e`` são frações de “tanque energético” em [0, 1]; consumo por viagem =
    KM_VIAGEM / autonomia_km(tipo_viagem).

    Os acumuladores cidade/estrada mantêm a decomposição para E[custo|p] = p*acum_c + (1-p)*acum_e
    quando ``prop_cidade == p`` (viagens i.i.d.).

    Retorna (acum_cidade, acum_estrada, snap_cidade, snap_estrada).
    """
    rng = np.random.default_rng(seed)
    U = rng.random((n_simulacoes, N_VIAGENS))

    acum_cidade = np.zeros(n_simulacoes, dtype=np.float64)
    acum_estrada = np.zeros(n_simulacoes, dtype=np.float64)
    elec_e = np.ones(n_simulacoes, dtype=np.float64)
    gas_e = np.ones(n_simulacoes, dtype=np.float64)
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
        tix = trip_num - 1
        is_c = U[:, tix] < float(prop_cidade)

        # --- Ramo elétrico ---
        idx_e = np.where(~modo_gas)[0]
        if idx_e.size:
            ja_viajou_eletrico = np.zeros(n_simulacoes, dtype=np.bool_)
            id_e0 = idx_e[elec_e[idx_e] <= 1e-14]
            if id_e0.size:
                aut_b = _auton_bat_vec(is_c[id_e0])
                elec_e[id_e0] = np.maximum(0.0, 1.0 - KM_VIAGEM / aut_b)
                _acumular_custo_viagem_eletrica(id_e0, acum_cidade, acum_estrada)
                ja_viajou_eletrico[id_e0] = True

            idx_e2 = idx_e[~ja_viajou_eletrico[idx_e]]
            if idx_e2.size:
                aut_b = _auton_bat_vec(is_c[idx_e2])
                km_left = elec_e[idx_e2] * aut_b
                maior_que_1 = km_left > KM_VIAGEM + 1e-9
                igual_fim = (~maior_que_1) & (elec_e[idx_e2] > 1e-15)
            else:
                maior_que_1 = np.array([], dtype=np.bool_)
                igual_fim = np.array([], dtype=np.bool_)

            id_gt1 = idx_e2[maior_que_1]
            if id_gt1.size:
                aut_bg = _auton_bat_vec(is_c[id_gt1])
                elec_e[id_gt1] = np.maximum(
                    0.0, elec_e[id_gt1] - KM_VIAGEM / aut_bg
                )
                _acumular_custo_viagem_eletrica(id_gt1, acum_cidade, acum_estrada)

            id_eq1 = idx_e2[igual_fim]
            if id_eq1.size:
                r = rng.random(id_eq1.size)
                esqueceu_recarga = r < P_ESQUECER_RECARGAR_BATERIA_ULTIMA_VIAGEM
                lembrou_recarga = ~esqueceu_recarga

                id_f = id_eq1[esqueceu_recarga]
                if id_f.size:
                    aut_g = _auton_gas_vec(is_c[id_f])
                    tem_gas = gas_e[id_f] * aut_g >= KM_VIAGEM - 1e-9
                    id_com_gas = id_f[tem_gas]
                    id_sem_gas = id_f[~tem_gas]
                    if id_com_gas.size:
                        modo_gas[id_com_gas] = True
                    if id_sem_gas.size:
                        r_ref = rng.random(id_sem_gas.size)
                        lembra_tanque = r_ref < P_LEMBRAR_ABASTECER_TANQUE_QUANDO_VAZIO
                        id_ab = id_sem_gas[lembra_tanque]
                        id_esq_tanque = id_sem_gas[~lembra_tanque]
                        if id_ab.size:
                            gas_e[id_ab] = 1.0
                            modo_gas[id_ab] = True
                        if id_esq_tanque.size:
                            aut_be = _auton_bat_vec(is_c[id_esq_tanque])
                            pode_ev = elec_e[id_esq_tanque] * aut_be >= KM_VIAGEM - 1e-9
                            id_ev = id_esq_tanque[pode_ev]
                            id_obrig = id_esq_tanque[~pode_ev]
                            if id_ev.size:
                                aut_bev = _auton_bat_vec(is_c[id_ev])
                                elec_e[id_ev] = np.maximum(0.0, elec_e[id_ev] - KM_VIAGEM / aut_bev)
                                _acumular_custo_viagem_eletrica(id_ev, acum_cidade, acum_estrada)
                            if id_obrig.size:
                                gas_e[id_obrig] = 1.0
                                modo_gas[id_obrig] = True

                id_l = id_eq1[lembrou_recarga]
                if id_l.size:
                    aut_bl = _auton_bat_vec(is_c[id_l])
                    elec_e[id_l] = np.maximum(0.0, 1.0 - KM_VIAGEM / aut_bl)
                    _acumular_custo_viagem_eletrica(id_l, acum_cidade, acum_estrada)

        idx_g = np.where(modo_gas)[0]
        if idx_g.size:
            aut_gg = _auton_gas_vec(is_c[idx_g])
            precisa = gas_e[idx_g] * aut_gg < KM_VIAGEM - 1e-9
            ids_sem = idx_g[precisa]
            if ids_sem.size:
                r_ref = rng.random(ids_sem.size)
                lembra_abastecer = r_ref < P_LEMBRAR_ABASTECER_TANQUE_QUANDO_VAZIO
                id_ab = ids_sem[lembra_abastecer]
                id_esq_t = ids_sem[~lembra_abastecer]
                if id_ab.size:
                    gas_e[id_ab] = 1.0
                if id_esq_t.size:
                    aut_be = _auton_bat_vec(is_c[id_esq_t])
                    pode_ev = elec_e[id_esq_t] * aut_be >= KM_VIAGEM - 1e-9
                    id_ev = id_esq_t[pode_ev]
                    id_obrig = id_esq_t[~pode_ev]
                    if id_ev.size:
                        aut_bev = _auton_bat_vec(is_c[id_ev])
                        elec_e[id_ev] = np.maximum(0.0, elec_e[id_ev] - KM_VIAGEM / aut_bev)
                        modo_gas[id_ev] = False
                        _acumular_custo_viagem_eletrica(id_ev, acum_cidade, acum_estrada)
                    if id_obrig.size:
                        gas_e[id_obrig] = 1.0

            idx_g_cons = np.where(modo_gas)[0]
            if idx_g_cons.size:
                aut_gc = _auton_gas_vec(is_c[idx_g_cons])
                gas_e[idx_g_cons] = np.maximum(
                    0.0, gas_e[idx_g_cons] - KM_VIAGEM / aut_gc
                )
                _acumular_custo_viagem_gasolina(idx_g_cons, acum_cidade, acum_estrada)
                r2 = rng.random(idx_g_cons.size)
                lembrou = r2 < P_LEMBRAR_RECARGAR_APOS_VIAGEM_GASOLINA
                id_lembrou = idx_g_cons[lembrou]
                if id_lembrou.size:
                    elec_e[id_lembrou] = 1.0
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


def caso2_curva_e_superficie_z2(
    props: np.ndarray,
    n_simulacoes: int,
    seed: int,
    com_snapshots: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Para cada proporção ``props[j]``, roda o Caso 2 com ``prop_cidade=props[j]`` (autonomia
    condicionada ao tipo de viagem). Retorna ``y2`` (custo esperado em 500k km) e ``Z2``
    (malha km × prop) se ``com_snapshots``.
    """
    props = np.asarray(props, dtype=np.float64)
    n_p = int(props.size)
    y2 = np.zeros(n_p, dtype=np.float64)
    n_marco = int(VIAGENS_NOS_MARCOS.size)
    Z2 = np.zeros((n_marco, n_p), dtype=np.float64)
    snaps = VIAGENS_NOS_MARCOS if com_snapshots else None
    for j in range(n_p):
        p = float(props[j])
        ac, ae, sc, se = simular_caso2_monte_carlo_vetorizado(
            n_simulacoes=n_simulacoes,
            seed=int(seed) + 17 * j,
            viagens_para_snapshots=snaps,
            prop_cidade=p,
        )
        y2[j] = float(np.mean(p * ac + (1.0 - p) * ae))
        if com_snapshots and sc is not None and se is not None:
            Z2[:, j] = np.mean(p * sc + (1.0 - p) * se, axis=0)
    return y2, Z2


def malha_superficies_3d(
    n_simulacoes: int = 4096,
    seed: int = 42,
    props: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monta as malhas 2D para três superfícies 3D: % cidade × km acumulado × custo (R$).

    ``props`` opcional (ex.: passo mais grosseiro no app interativo). Caso 2: uma simulação
    por coluna de ``props`` com ``prop_cidade`` igual àquela proporção.

    Retorna (X_pct, Y_km, Z1, Z2, Z3).
    """
    if props is None:
        props = np.round(np.arange(0.01, 1.01, 0.01), 2)
    else:
        props = np.asarray(props, dtype=np.float64)
    eixo_x_pct = props * 100.0
    X_3d, Y_3d = np.meshgrid(eixo_x_pct, KM_MARCOS_ACUM.astype(np.float64))
    km_col = KM_MARCOS_ACUM[:, np.newaxis].astype(np.float64)
    prop_row = props[np.newaxis, :].astype(np.float64)
    Z1_3d = custo_caso1_acumulado_ate_km(km_col, prop_row)

    _y2_tmp, Z2_3d = caso2_curva_e_superficie_z2(
        props, n_simulacoes=n_simulacoes, seed=seed, com_snapshots=True
    )
    Z3_3d = custo_caso3_acumulado_ate_km(km_col, prop_row)
    return X_3d, Y_3d, Z1_3d, Z2_3d, Z3_3d


def pontos_intersecao_par_superficies(
    X: np.ndarray,
    Y: np.ndarray,
    Za: np.ndarray,
    Zb: np.ndarray,
) -> np.ndarray:
    """
    Pontos 3D onde duas superfícies discretas se cruzam, aproximados nas arestas da malha
    (interpolação linear onde (Za - Zb) muda de sinal).
    Colunas de saída: x (% cidade), y (km), z (R$).
    """
    ny, nx = Za.shape
    pts: list[tuple[float, float, float]] = []
    D = Za - Zb

    def aresta(d0: float, d1: float, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> None:
        if np.isnan(d0) or np.isnan(d1):
            return
        if abs(d0) < 1e-9:
            pts.append((float(x0), float(y0), float(z0)))
            return
        if d0 * d1 > 0:
            return
        if abs(d1 - d0) < 1e-18:
            return
        t = -d0 / (d1 - d0)
        if not (0.0 <= t <= 1.0):
            return
        x = (1.0 - t) * x0 + t * x1
        y = (1.0 - t) * y0 + t * y1
        z = (1.0 - t) * z0 + t * z1
        pts.append((x, y, z))

    for i in range(ny):
        for j in range(nx - 1):
            aresta(
                D[i, j],
                D[i, j + 1],
                X[i, j],
                X[i, j + 1],
                Y[i, j],
                Y[i, j + 1],
                Za[i, j],
                Za[i, j + 1],
            )
    for i in range(ny - 1):
        for j in range(nx):
            aresta(
                D[i, j],
                D[i + 1, j],
                X[i, j],
                X[i + 1, j],
                Y[i, j],
                Y[i + 1, j],
                Za[i, j],
                Za[i + 1, j],
            )

    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def pontos_cruzamento_triplo(
    Z1: np.ndarray,
    Z2: np.ndarray,
    Z3: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    atol: float | None = None,
) -> np.ndarray:
    """Vértices onde |Z1-Z2|, |Z2-Z3| e |Z1-Z3| são simultaneamente pequenos."""
    span = max(float(np.ptp(Z1)), float(np.ptp(Z2)), float(np.ptp(Z3)), 1.0)
    if atol is None:
        atol = 0.02 * span
    m = (np.abs(Z1 - Z2) <= atol) & (np.abs(Z2 - Z3) <= atol) & (np.abs(Z1 - Z3) <= atol)
    if not np.any(m):
        return np.zeros((0, 3), dtype=np.float64)
    return np.column_stack([X[m], Y[m], ((Z1[m] + Z2[m] + Z3[m]) / 3.0)])


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

    # Caso 2: uma simulação por proporção p (autonomia gas/elétrico × cidade/estrada)
    print(
        "Monte Carlo Caso 2: 100 proporções × marcos de km (autonomia 600/750 gas, 250/150 elétrico)…"
    )
    y2, Z2_3d = caso2_curva_e_superficie_z2(
        props,
        n_simulacoes=4096,
        seed=42,
        com_snapshots=True,
    )

    ac0, ae0, _, _ = simular_caso2_monte_carlo_vetorizado(
        n_simulacoes=4096,
        seed=42,
        viagens_para_snapshots=None,
        prop_cidade=0.5,
    )
    custo_ref_p50 = 0.5 * ac0 + 0.5 * ae0
    dp_c2 = float(np.std(custo_ref_p50, ddof=1)) if ac0.size > 1 else 0.0

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
        "Comparação de custos de abastecimento — híbrido (500.000 km, viagens de 25 km)",
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

    P12 = pontos_intersecao_par_superficies(X_3d, Y_3d, Z1_3d, Z2_3d)
    P13 = pontos_intersecao_par_superficies(X_3d, Y_3d, Z1_3d, Z3_3d)
    P23 = pontos_intersecao_par_superficies(X_3d, Y_3d, Z2_3d, Z3_3d)
    Pt3 = pontos_cruzamento_triplo(Z1_3d, Z2_3d, Z3_3d, X_3d, Y_3d)

    cruz_handles: list[Line2D] = []
    if P12.size:
        ax3d.scatter(
            P12[:, 0],
            P12[:, 1],
            P12[:, 2],
            s=16,
            c="#14532d",
            edgecolors=_COR_MARCA_EIXO,
            linewidths=0.5,
            depthshade=True,
        )
        for p in P12:
            ax3d.text(
                p[0],
                p[1],
                p[2],
                f"  {p[2]:,.0f}".replace(",", "."),
                fontsize=5,
                color=_COR_ROTULO_EIXO,
                clip_on=True,
            )
        cruz_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                color="w",
                markerfacecolor="#14532d",
                markeredgecolor=_COR_MARCA_EIXO,
                markersize=6,
                label="Cruz.: 1 × 2",
            )
        )
    if P13.size:
        ax3d.scatter(
            P13[:, 0],
            P13[:, 1],
            P13[:, 2],
            s=16,
            c="#0c4a6e",
            edgecolors=_COR_MARCA_EIXO,
            linewidths=0.5,
            depthshade=True,
        )
        for p in P13:
            ax3d.text(
                p[0],
                p[1],
                p[2],
                f"  {p[2]:,.0f}".replace(",", "."),
                fontsize=5,
                color=_COR_ROTULO_EIXO,
                clip_on=True,
            )
        cruz_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                color="w",
                markerfacecolor="#0c4a6e",
                markeredgecolor=_COR_MARCA_EIXO,
                markersize=6,
                label="Cruz.: 1 × 3",
            )
        )
    if P23.size:
        ax3d.scatter(
            P23[:, 0],
            P23[:, 1],
            P23[:, 2],
            s=16,
            c="#831843",
            edgecolors=_COR_MARCA_EIXO,
            linewidths=0.5,
            depthshade=True,
        )
        for p in P23:
            ax3d.text(
                p[0],
                p[1],
                p[2],
                f"  {p[2]:,.0f}".replace(",", "."),
                fontsize=5,
                color=_COR_ROTULO_EIXO,
                clip_on=True,
            )
        cruz_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                color="w",
                markerfacecolor="#831843",
                markeredgecolor=_COR_MARCA_EIXO,
                markersize=6,
                label="Cruz.: 2 × 3",
            )
        )
    if Pt3.size:
        ax3d.scatter(
            Pt3[:, 0],
            Pt3[:, 1],
            Pt3[:, 2],
            s=36,
            c="#f59e0b",
            edgecolors="#78350f",
            linewidths=0.6,
            depthshade=True,
        )
        for p in Pt3:
            ax3d.text(
                p[0],
                p[1],
                p[2],
                f"  {p[2]:,.0f}".replace(",", "."),
                fontsize=6,
                color="#78350f",
                clip_on=True,
            )
        cruz_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                color="w",
                markerfacecolor="#f59e0b",
                markeredgecolor="#78350f",
                markersize=7,
                label="Cruz.: 1 × 2 × 3",
            )
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
    leg3d = ax3d.legend(
        handles=leg + cruz_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )
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
