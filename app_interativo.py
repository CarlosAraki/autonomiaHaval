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

# Posições de texto alternadas para reduzir sobreposição nos cruzamentos 3D
_TEXTPOS_CRUZAMENTO = [
    "top center",
    "top left",
    "top right",
    "bottom center",
    "bottom left",
    "bottom right",
    "middle left",
    "middle right",
]

# Rotulos da sidebar -> chaves internas dos tres casos
_ROTULOS_PLANOS: dict[str, str] = {
    "Caso 1 — Baseline": "c1",
    "Caso 2 — Comportamental (MC)": "c2",
    "Caso 3 — Inteligente": "c3",
}

# Presets de câmera (scene.camera): chave exibida na UI
_CAMERA_PRESETS: dict[str, dict] = {
    "Isométrica": {"eye": {"x": 1.55, "y": -1.45, "z": 0.85}},
    "Cidade × km (vista de cima)": {"eye": {"x": 0, "y": 0, "z": 2.75}, "up": {"x": 0, "y": 1, "z": 0}},
    "Cidade × custo (vista lateral)": {
        "eye": {"x": 0, "y": -2.75, "z": 0.4},
        "up": {"x": 0, "y": 0, "z": 1},
    },
    "Km × custo (vista lateral)": {
        "eye": {"x": 2.75, "y": 0, "z": 0.4},
        "up": {"x": 0, "y": 0, "z": 1},
    },
    "Enfatizar custo (Z)": {"eye": {"x": 0.35, "y": -0.35, "z": 1.85}},
}


def _estilo_espacamento_rotulos(chave: str) -> tuple[int, float, int]:
    """(textfont_size, marker_size, marker_linewidth)."""
    if chave == "Compacto":
        return 8, 5.5, 1
    if chave == "Amplo":
        return 12, 11.0, 2
    return 10, 8.0, 2  # Normal


def _botoes_camera_plotly() -> list[dict]:
    """Botões relayout para o menu de vista no próprio gráfico."""
    out: list[dict] = []
    for label, cam in _CAMERA_PRESETS.items():
        out.append(
            dict(
                label=label,
                method="relayout",
                args=[{"scene.camera": cam}],
            )
        )
    return out

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
    cruzamentos: dict[str, bool] | None = None,
    espacamento_rotulos: str = "Normal",
    vista_camera: str = "Isométrica",
) -> go.Figure:
    fig = go.Figure()

    fs_txt, fz_mrk, lw_mrk = _estilo_espacamento_rotulos(espacamento_rotulos)

    if cruzamentos is None:
        cruzamentos = {"pair_12": True, "pair_13": True, "pair_23": True, "triple": True}

    specs: list[tuple[str, np.ndarray, str, str]] = [
        ("c1", Z1, "Caso 1 — Ciclo fixo (baseline)", "#2ecc71"),
        ("c2", Z2, "Caso 2 — Elétrico + esquecimento (MC)", "#e74c3c"),
        ("c3", Z3, "Caso 3 — Chaveamento inteligente", "#3498db"),
    ]

    primeiro_plano = True
    for key, Z, nome, cor in specs:
        if not mostrar.get(key, True):
            continue
        trace_kw: dict = dict(
            x=X,
            y=Y,
            z=Z,
            name=nome,
            opacity=0.52,
            colorscale=[[0.0, cor], [1.0, cor]],
            surfacecolor=Z,
            showscale=False,
            lighting=dict(ambient=0.65, diffuse=0.85, specular=0.25),
            legendgroup="planos",
            hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Cidade: %{x:.1f} %<br>"
                    "Km acumulado: %{y:,.0f}<br>"
                    "Custo: %{z:,.0f} R$<extra></extra>"
                ),
        )
        if primeiro_plano:
            trace_kw["legendgrouptitle_text"] = "Superfícies"
            primeiro_plano = False
        fig.add_trace(go.Surface(**trace_kw))

    if mostrar_cruzamentos:
        pares: list[tuple[str, str, np.ndarray, str, str]] = []
        if (
            mostrar.get("c1", True)
            and mostrar.get("c2", True)
            and cruzamentos.get("pair_12", True)
        ):
            p12 = pontos_intersecao_par_superficies(X, Y, Z1, Z2)
            pares.append(("12", "Cruz.: Caso 1 × Caso 2", p12, "#14532d", "diamond-open"))
        if (
            mostrar.get("c1", True)
            and mostrar.get("c3", True)
            and cruzamentos.get("pair_13", True)
        ):
            p13 = pontos_intersecao_par_superficies(X, Y, Z1, Z3)
            pares.append(("13", "Cruz.: Caso 1 × Caso 3", p13, "#0c4a6e", "square-open"))
        if (
            mostrar.get("c2", True)
            and mostrar.get("c3", True)
            and cruzamentos.get("pair_23", True)
        ):
            p23 = pontos_intersecao_par_superficies(X, Y, Z2, Z3)
            # Scatter3d só aceita: circle, circle-open, cross, diamond, diamond-open, square, square-open, x
            pares.append(("23", "Cruz.: Caso 2 × Caso 3", p23, "#831843", "cross"))

        km_ref = float(km_teto_cal) if km_teto_cal > 0 else float(np.nanmax(Y)) if Y.size else 1.0

        primeiro_cruz = True
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
            n_pts = len(P)
            posicoes = [_TEXTPOS_CRUZAMENTO[i % len(_TEXTPOS_CRUZAMENTO)] for i in range(n_pts)]
            modo = "markers+text" if rotulos_data_picker else "markers"
            sc_kw: dict = dict(
                x=P[:, 0],
                y=P[:, 1],
                z=P[:, 2],
                mode=modo,
                name=legenda,
                text=textos,
                textposition=posicoes,
                textfont=dict(size=fs_txt, color=_COR_EIXO_TITULO, family="Arial, sans-serif"),
                marker=dict(
                    size=fz_mrk,
                    color=cor_marca,
                    symbol=simbolo,
                    line=dict(width=lw_mrk, color=_COR_EIXO_TITULO),
                ),
                hovertemplate="%{text}<extra></extra>",
                legendgroup="cruz",
            )
            if primeiro_cruz:
                sc_kw["legendgrouptitle_text"] = "Cruzamentos"
                primeiro_cruz = False
            fig.add_trace(go.Scatter3d(**sc_kw))

        if (
            mostrar.get("c1")
            and mostrar.get("c2")
            and mostrar.get("c3")
            and cruzamentos.get("triple", True)
        ):
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
                pos_t = [_TEXTPOS_CRUZAMENTO[i % len(_TEXTPOS_CRUZAMENTO)] for i in range(len(Pt))]
                fs_tri = min(fs_txt + 1, 14)
                mk_tri = min(fz_mrk + 1.5, 14.0)
                tri_kw: dict = dict(
                    x=Pt[:, 0],
                    y=Pt[:, 1],
                    z=Pt[:, 2],
                    mode=modo_t,
                    name="Cruz.: os 3 planos",
                    text=ttxt,
                    textposition=pos_t,
                    textfont=dict(size=fs_tri, color="#422006", family="Arial, sans-serif"),
                    marker=dict(
                        size=mk_tri,
                        color="#f59e0b",
                        symbol="circle",
                        line=dict(width=lw_mrk, color="#78350f"),
                    ),
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup="cruz",
                )
                if primeiro_cruz:
                    tri_kw["legendgrouptitle_text"] = "Cruzamentos"
                fig.add_trace(go.Scatter3d(**tri_kw))

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

    cam_inicial = dict(_CAMERA_PRESETS.get(vista_camera, next(iter(_CAMERA_PRESETS.values()))))
    chaves_cam = list(_CAMERA_PRESETS.keys())
    try:
        idx_menu_cam = chaves_cam.index(vista_camera)
    except ValueError:
        idx_menu_cam = 0

    fig.update_layout(
        title=dict(
            text="Custo acumulado (R$) × proporção cidade × km percorridos",
            x=0.5,
            xanchor="center",
            font=dict(color=_COR_EIXO_TITULO, size=17, family="Arial, sans-serif"),
        ),
        height=altura_px,
        margin=dict(l=0, r=0, t=72, b=0),
        uirevision="custo3d",
        scene=dict(
            xaxis=_eixo_3d("Proporção de uso na cidade (%)"),
            yaxis=_eixo_3d("Utilização acumulada (km)"),
            zaxis=_eixo_3d("Custo acumulado (R$)"),
            bgcolor="rgb(248,249,250)",
            aspectmode="cube",
            camera=cam_inicial,
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(15,23,42,0.12)",
            borderwidth=1,
            tracegroupgap=16,
            font=dict(color=_COR_EIXO_TICK, size=12, family="Arial, sans-serif"),
            title=dict(
                text="<b>Legenda</b> — clique para mostrar ou ocultar cada série",
                font=dict(size=11, color=_COR_EIXO_TICK, family="Arial, sans-serif"),
            ),
            itemsizing="constant",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        annotations=[
            dict(
                text="<b>Vista (eixos)</b>",
                x=0.98,
                y=1.028,
                xref="paper",
                yref="paper",
                xanchor="right",
                showarrow=False,
                font=dict(size=12, color=_COR_EIXO_TITULO, family="Arial, sans-serif"),
            )
        ],
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                active=idx_menu_cam,
                x=0.99,
                xanchor="right",
                y=1.0,
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor=_COR_EIXO_TICK,
                borderwidth=1,
                font=dict(family="Arial, sans-serif", color=_COR_EIXO_TITULO, size=11),
                buttons=_botoes_camera_plotly(),
            )
        ],
    )
    return fig


def main() -> None:
    st.title("Custo de abastecimento — visualização 3D interativa")
    st.caption(
        "Gire, dê zoom e arraste com o mouse. Use a **legenda** do gráfico para ligar ou desligar cada "
        "superfície e cada cruzamento; no canto superior direito, o menu **Vista (eixos)** muda o ângulo "
        "sem recalcular a malha. A barra de ferramentas do Plotly permite exportar imagem e resetar a câmera."
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
        st.subheader("Legenda — planos (superfícies)")
        planos_sel = st.multiselect(
            "Quais superfícies exibir",
            options=list(_ROTULOS_PLANOS.keys()),
            default=list(_ROTULOS_PLANOS.keys()),
            help="Define quais planos entram no 3D. Você também pode alternar pelos itens da legenda do gráfico.",
        )
        st.divider()
        st.subheader("Legenda — cruzamentos")
        mostrar_cruz = st.checkbox(
            "Mostrar pontos onde as superfícies se cruzam",
            value=True,
            help="Interseções aproximadas na malha; escolha abaixo quais pares exibir.",
        )
        rotulos_dp = st.checkbox(
            "Rótulos com data do período em cada cruzamento",
            value=True,
            help="Cada ponto mostra data estimada (km alinhado ao intervalo de datas), % cidade, km e R$.",
        )
        espacamento = st.select_slider(
            "Espaçamento dos rótulos nos cruzamentos",
            options=["Compacto", "Normal", "Amplo"],
            value="Normal",
            help="Amplo: fonte e marcadores maiores; posições de texto alternadas para reduzir sobreposição.",
        )

        mostrar = {v: (k in planos_sel) for k, v in _ROTULOS_PLANOS.items()}
        if not any(mostrar.values()):
            st.warning("Selecione pelo menos um plano.")
            mostrar = {v: True for v in _ROTULOS_PLANOS.values()}

        disponiveis: list[tuple[str, str]] = []
        if mostrar["c1"] and mostrar["c2"]:
            disponiveis.append(("Caso 1 x Caso 2", "pair_12"))
        if mostrar["c1"] and mostrar["c3"]:
            disponiveis.append(("Caso 1 x Caso 3", "pair_13"))
        if mostrar["c2"] and mostrar["c3"]:
            disponiveis.append(("Caso 2 x Caso 3", "pair_23"))
        if mostrar["c1"] and mostrar["c2"] and mostrar["c3"]:
            disponiveis.append(("Tres planos ao mesmo tempo (triplo)", "triple"))

        cruz_dict: dict[str, bool] | None = None
        if mostrar_cruz and disponiveis:
            labels_disp = [x[0] for x in disponiveis]
            cruz_sel = st.multiselect(
                "Quais cruzamentos exibir",
                options=labels_disp,
                default=labels_disp,
                help="Reduza a poluição visual mostrando só os cruzamentos que interessam.",
            )
            cruz_dict = {k: False for k in ("pair_12", "pair_13", "pair_23", "triple")}
            for lbl, key in disponiveis:
                if lbl in cruz_sel:
                    cruz_dict[key] = True
        elif mostrar_cruz and not disponiveis:
            st.caption("Ative pelo menos dois planos para haver cruzamentos.")

        st.divider()
        st.subheader("Vista entre eixos (câmera)")
        vista_inicial = st.selectbox(
            "Ângulo inicial do gráfico 3D",
            options=list(_CAMERA_PRESETS.keys()),
            index=0,
            help="Define a vista ao recarregar a página. No gráfico, use o menu no canto superior direito para trocar sem recalcular.",
        )

    X, Y, Z1, Z2, Z3 = carregar_malha(n_simulacoes=n_sim, seed=seed)
    Xs, Ys, Z1s, Z2s, Z3s = recortar_por_km_max(X, Y, Z1, Z2, Z3, km_teto)

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
        cruzamentos=cruz_dict,
        espacamento_rotulos=espacamento,
        vista_camera=vista_inicial,
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
