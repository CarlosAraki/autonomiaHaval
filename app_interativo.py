#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Painel interativo (Streamlit + Plotly) — superfícies 3D de custo acumulado.

Execute: streamlit run app_interativo.py
"""

from __future__ import annotations

from datetime import date

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from otimizacao_abastecimento_hibrido import KM_MARCOS_ACUM, malha_superficies_3d

# Contraste para títulos de eixos e marcas (Plotly)
_COR_EIXO_TITULO = "#0f172a"  # slate-900
_COR_EIXO_TICK = "#1e293b"  # slate-800

st.set_page_config(
    page_title="Híbrido — custo 3D",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }
    div[data-testid="stPlotlyChart"] > div { min-height: inherit; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Calculando malha 3D e Monte Carlo (Caso 2)…")
def carregar_malha(n_simulacoes: int, seed: int) -> tuple:
    return malha_superficies_3d(n_simulacoes=n_simulacoes, seed=seed)


def recortar_por_km_max(
    X: np.ndarray,
    Y: np.ndarray,
    Z1: np.ndarray,
    Z2: np.ndarray,
    Z3: np.ndarray,
    km_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mantém apenas marcos de km ≤ km_max (pelo menos uma linha)."""
    km_max = float(min(km_max, 500_000.0))
    idx = KM_MARCOS_ACUM <= km_max
    if not np.any(idx):
        idx = np.zeros_like(idx, dtype=bool)
        idx[0] = True
    return X[idx], Y[idx], Z1[idx], Z2[idx], Z3[idx]


def figura_superficies(
    X: np.ndarray,
    Y: np.ndarray,
    Z1: np.ndarray,
    Z2: np.ndarray,
    Z3: np.ndarray,
    *,
    altura_px: int,
    mostrar: dict[str, bool],
) -> go.Figure:
    fig = go.Figure()

    specs: list[tuple[str, np.ndarray, str, str]] = [
        ("c1", Z1, "Caso 1 — Ciclo fixo (baseline)", "#2ecc71"),
        ("c2", Z2, "Caso 2 — Elétrico + esquecimento (MC)", "#e74c3c"),
        ("c3", Z3, "Caso 3 — Chaveamento inteligente", "#3498db"),
    ]

    for key, Z, nome, cor in specs:
        if not mostrar.get(key, True):
            continue
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                name=nome,
                opacity=0.52,
                colorscale=[[0.0, cor], [1.0, cor]],
                surfacecolor=Z,
                showscale=False,
                lighting=dict(ambient=0.65, diffuse=0.85, specular=0.25),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Cidade: %{x:.1f} %<br>"
                    "Km acumulado: %{y:,.0f}<br>"
                    "Custo: %{z:,.0f} R$<extra></extra>"
                ),
            )
        )

    _font_titulo_eixo = dict(color=_COR_EIXO_TITULO, size=13, family="Arial, sans-serif")
    _font_tick_eixo = dict(color=_COR_EIXO_TICK, size=11, family="Arial, sans-serif")

    def _eixo_3d(titulo: str) -> dict:
        return {
            "title": {"text": titulo, "font": dict(_font_titulo_eixo)},
            "tickfont": dict(_font_tick_eixo),
            "showbackground": True,
            "backgroundcolor": "rgb(230,230,230)",
            "gridcolor": "white",
            "linecolor": _COR_EIXO_TICK,
            "linewidth": 1,
            "tickcolor": _COR_EIXO_TICK,
        }

    fig.update_layout(
        title=dict(
            text="Custo acumulado (R$) × proporção cidade × km percorridos",
            x=0.5,
            xanchor="center",
            font=dict(color=_COR_EIXO_TITULO, size=17, family="Arial, sans-serif"),
        ),
        height=altura_px,
        margin=dict(l=0, r=0, t=56, b=0),
        scene=dict(
            xaxis=_eixo_3d("Proporção de uso na cidade (%)"),
            yaxis=_eixo_3d("Utilização acumulada (km)"),
            zaxis=_eixo_3d("Custo acumulado (R$)"),
            bgcolor="rgb(248,249,250)",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.55, y=-1.45, z=0.85)),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.75)",
            font=dict(color=_COR_EIXO_TICK, size=12, family="Arial, sans-serif"),
        ),
    )
    return fig


def main() -> None:
    st.title("Custo de abastecimento — visualização 3D interativa")
    st.caption(
        "Gire, dê zoom e arraste com o mouse. Use a barra de ferramentas do gráfico "
        "para captura de tela e reset de câmera."
    )

    with st.sidebar:
        st.header("Simulação (Caso 2)")
        n_sim = st.select_slider(
            "Réplicas Monte Carlo",
            options=[512, 1024, 2048, 4096],
            value=4096,
            help="Mais réplicas: média mais estável, cálculo mais lento.",
        )
        seed = int(st.number_input("Seed aleatória", value=42, step=1))

        st.divider()
        st.header("Período em calendário → km exibidos")
        st.caption(
            "Converte o intervalo entre duas datas em um teto de **km acumulados** no eixo Y "
            "(útil para enquadrar a vida útil em uma linha do tempo). Limitado a 500.000 km."
        )
        hoje = date.today()
        d_ini = st.date_input(
            "Data inicial",
            value=date(hoje.year - 5, 1, 1),
            help="Início do período de referência (visualização).",
        )
        d_fim = st.date_input(
            "Data final",
            value=hoje,
            help="Fim do período de referência (visualização).",
        )
        if d_fim < d_ini:
            st.warning("A data final é anterior à inicial; invertendo para o cálculo.")
            d_ini, d_fim = d_fim, d_ini
        dias = max(1, (d_fim - d_ini).days)
        km_por_dia = st.slider(
            "Km médios por dia",
            min_value=5,
            max_value=500,
            value=137,
            help="Km acumulado máximo mostrado = min(500.000, dias × km/dia).",
        )
        km_teto = min(500_000.0, float(dias * km_por_dia))
        st.metric("Teto de km no gráfico", f"{km_teto:,.0f} km".replace(",", "."))

        st.divider()
        st.header("Visualização ampla")
        altura = st.slider(
            "Altura do gráfico (px)",
            min_value=480,
            max_value=1400,
            value=960,
            step=20,
            help="Aumente para ocupar melhor monitores grandes ou modo apresentação.",
        )
        st.caption("A página usa layout **wide**; o gráfico expande à largura útil.")

        st.divider()
        st.subheader("Camadas")
        c1 = st.checkbox("Caso 1 — Baseline", value=True)
        c2 = st.checkbox("Caso 2 — Comportamental (MC)", value=True)
        c3 = st.checkbox("Caso 3 — Inteligente", value=True)

    X, Y, Z1, Z2, Z3 = carregar_malha(n_simulacoes=n_sim, seed=seed)
    Xs, Ys, Z1s, Z2s, Z3s = recortar_por_km_max(X, Y, Z1, Z2, Z3, km_teto)

    mostrar = {"c1": c1, "c2": c2, "c3": c3}
    if not any(mostrar.values()):
        st.warning("Marque pelo menos um caso na barra lateral.")
        mostrar = {k: True for k in mostrar}

    fig = figura_superficies(
        Xs, Ys, Z1s, Z2s, Z3s, altura_px=altura, mostrar=mostrar
    )

    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "custo_hibrido_3d",
            "height": altura,
            "width": 1600,
            "scale": 2,
        },
    }
    st.plotly_chart(fig, use_container_width=True, config=config)

    with st.expander("Notas metodológicas"):
        st.markdown(
            """
- **Eixo X:** proporção de km em cidade (1% a 100%).
- **Eixo Y:** km acumulados em marcos de 5.000 km (recortados pelo teto derivado do calendário).
- **Eixo Z:** custo acumulado em R$ (rendimento gasolina/bateria × cidade/estrada conforme o script principal).
- O **datepicker** não altera o modelo estatístico: apenas define até quantos km a malha é exibida, via `dias × km/dia`.
- Caso 2: esperança sobre rotulagem cidade/estrada com probabilidade *p* (linear); uma simulação cobre todo o grid de *p*.
            """
        )


if __name__ == "__main__":
    main()
