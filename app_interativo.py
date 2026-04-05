#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Painel interativo (Streamlit + Plotly) — superfícies 3D de custo acumulado.

Execute: streamlit run app_interativo.py
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from otimizacao_abastecimento_hibrido import (
    KM_MARCOS_ACUM,
    malha_superficies_3d,
    pontos_cruzamento_triplo,
    pontos_intersecao_par_superficies,
)

# Malha 3D mais esparsa no app (20 colunas) para não rodar 100× Monte Carlo a cada refresh
_PROPS_MALHA_APP = np.round(np.arange(0.05, 1.01, 0.05), 2)

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


@st.cache_data(show_spinner="Calculando malha 3D e Monte Carlo (Caso 2, várias proporções)…")
def carregar_malha(n_simulacoes: int, seed: int) -> tuple:
    return malha_superficies_3d(
        n_simulacoes=n_simulacoes,
        seed=seed,
        props=_PROPS_MALHA_APP,
    )


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


def _data_no_periodo(
    y_km: float,
    d_ini: date,
    d_fim: date,
    km_referencia: float,
) -> date:
    """Alinha km acumulado ao intervalo do date picker (proporção linear)."""
    if km_referencia <= 0:
        return d_ini
    frac = float(np.clip(y_km / km_referencia, 0.0, 1.0))
    nd = max(0, (d_fim - d_ini).days)
    return d_ini + timedelta(days=int(round(frac * nd)))


def _rotulo_cruzamento(
    px: float,
    py: float,
    pz: float,
    *,
    d_ini: date,
    d_fim: date,
    km_ref: float,
    incluir_data: bool,
) -> str:
    r_br = f"{pz:,.0f}".replace(",", ".")
    base = f"{px:.0f}% cidade<br>{py:,.0f} km<br>R$ {r_br}".replace(",", ".")
    if incluir_data and km_ref > 0:
        dt = _data_no_periodo(py, d_ini, d_fim, km_ref)
        base = f"{dt:%d/%m/%Y}<br>" + base
    return base


def figura_superficies(
    X: np.ndarray,
    Y: np.ndarray,
    Z1: np.ndarray,
    Z2: np.ndarray,
    Z3: np.ndarray,
    *,
    altura_px: int,
    mostrar: dict[str, bool],
    d_ini_cal: date,
    d_fim_cal: date,
    km_teto_cal: float,
    mostrar_cruzamentos: bool = True,
    rotulos_data_picker: bool = True,
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

    if mostrar_cruzamentos:
        pares: list[tuple[str, str, np.ndarray, str, str]] = []
        if mostrar.get("c1", True) and mostrar.get("c2", True):
            p12 = pontos_intersecao_par_superficies(X, Y, Z1, Z2)
            pares.append(("12", "Cruz.: Caso 1 × Caso 2", p12, "#14532d", "diamond-open"))
        if mostrar.get("c1", True) and mostrar.get("c3", True):
            p13 = pontos_intersecao_par_superficies(X, Y, Z1, Z3)
            pares.append(("13", "Cruz.: Caso 1 × Caso 3", p13, "#0c4a6e", "square-open"))
        if mostrar.get("c2", True) and mostrar.get("c3", True):
            p23 = pontos_intersecao_par_superficies(X, Y, Z2, Z3)
            # Scatter3d só aceita: circle, circle-open, cross, diamond, diamond-open, square, square-open, x
            pares.append(("23", "Cruz.: Caso 2 × Caso 3", p23, "#831843", "cross"))

        km_ref = float(km_teto_cal) if km_teto_cal > 0 else float(np.nanmax(Y)) if Y.size else 1.0

        for _key, legenda, P, cor_marca, simbolo in pares:
            if P.size == 0:
                continue
            textos = [
                _rotulo_cruzamento(
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    d_ini=d_ini_cal,
                    d_fim=d_fim_cal,
                    km_ref=km_ref,
                    incluir_data=rotulos_data_picker,
                )
                for p in P
            ]
            modo = "markers+text" if rotulos_data_picker else "markers"
            fig.add_trace(
                go.Scatter3d(
                    x=P[:, 0],
                    y=P[:, 1],
                    z=P[:, 2],
                    mode=modo,
                    name=legenda,
                    text=textos,
                    textposition="top center",
                    textfont=dict(size=9, color=_COR_EIXO_TITULO, family="Arial, sans-serif"),
                    marker=dict(size=6, color=cor_marca, symbol=simbolo, line=dict(width=1, color=_COR_EIXO_TITULO)),
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup="cruz",
                )
            )

        if mostrar.get("c1") and mostrar.get("c2") and mostrar.get("c3"):
            Pt = pontos_cruzamento_triplo(Z1, Z2, Z3, X, Y)
            if Pt.size > 0:
                ttxt = [
                    _rotulo_cruzamento(
                        float(p[0]),
                        float(p[1]),
                        float(p[2]),
                        d_ini=d_ini_cal,
                        d_fim=d_fim_cal,
                        km_ref=km_ref,
                        incluir_data=rotulos_data_picker,
                    )
                    for p in Pt
                ]
                modo_t = "markers+text" if rotulos_data_picker else "markers"
                fig.add_trace(
                    go.Scatter3d(
                        x=Pt[:, 0],
                        y=Pt[:, 1],
                        z=Pt[:, 2],
                        mode=modo_t,
                        name="Cruz.: os 3 planos",
                        text=ttxt,
                        textposition="top center",
                        textfont=dict(size=10, color="#422006", family="Arial, sans-serif"),
                        marker=dict(size=9, color="#f59e0b", symbol="circle", line=dict(width=1, color="#78350f")),
                        hovertemplate="%{text}<extra></extra>",
                        legendgroup="cruz",
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

        st.divider()
        st.subheader("Cruzamentos entre planos")
        mostrar_cruz = st.checkbox(
            "Mostrar pontos onde as superfícies se cruzam",
            value=True,
            help="Marca interseções aproximadas (malha) e entradas na legenda.",
        )
        rotulos_dp = st.checkbox(
            "Rótulos com data do período (datepicker) em cada cruzamento",
            value=True,
            help="Cada ponto mostra data estimada (km alinhado ao intervalo de datas), % cidade, km e R$.",
        )

    X, Y, Z1, Z2, Z3 = carregar_malha(n_simulacoes=n_sim, seed=seed)
    Xs, Ys, Z1s, Z2s, Z3s = recortar_por_km_max(X, Y, Z1, Z2, Z3, km_teto)

    mostrar = {"c1": c1, "c2": c2, "c3": c3}
    if not any(mostrar.values()):
        st.warning("Marque pelo menos um caso na barra lateral.")
        mostrar = {k: True for k in mostrar}

    fig = figura_superficies(
        Xs,
        Ys,
        Z1s,
        Z2s,
        Z3s,
        altura_px=altura,
        mostrar=mostrar,
        d_ini_cal=d_ini,
        d_fim_cal=d_fim,
        km_teto_cal=km_teto,
        mostrar_cruzamentos=mostrar_cruz,
        rotulos_data_picker=rotulos_dp,
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
- O **datepicker** não altera o modelo estatístico: define o teto de km na malha (`dias × km/dia`) e, nos **cruzamentos**, a **data exibida** no rótulo (interpolação linear entre data inicial e final conforme o km do ponto).
- Caso 2: esperança sobre rotulagem cidade/estrada com probabilidade *p* (linear); uma simulação cobre todo o grid de *p*.
            """
        )


if __name__ == "__main__":
    main()
