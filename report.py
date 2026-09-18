"""
report.py
Capa de visualización, desacoplada del cálculo. No conoce RiskEngine, ni
backtester, ni optimizer — solo recibe estructuras de datos ya calculadas
(MetricasBasicas, lista de TradeFuturo, ResultadoMonteCarlo) y las pinta.
Esto permite testear metrics.py/backtester.py sin nunca levantar Streamlit.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from metrics import MetricasBasicas, ResultadoMonteCarlo


def render_kpis(metricas: MetricasBasicas) -> None:
    """4 columnas de st.metric: Win Rate, Profit Factor, Total Trades, EV."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win Rate", f"{metricas.win_rate_pct:.1f}%")

    pf_texto = "∞" if metricas.profit_factor == float("inf") else f"{metricas.profit_factor:.2f}"
    col2.metric("Profit Factor", pf_texto)

    col3.metric("Total Trades", f"{metricas.total_trades}")
    col4.metric("EV por Trade", f"${metricas.ev_por_trade:,.2f}")


def plot_equity_curve(trades: List) -> None:
    """Evolución de `equity_tras_cierre` de cada TradeFuturo en el tiempo."""
    if not trades:
        st.info("Sin trades para graficar.")
        return

    df = pd.DataFrame({
        "fecha_salida": [t.fecha_salida for t in trades],
        "equity": [t.equity_tras_cierre for t in trades],
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["fecha_salida"], y=df["equity"],
        mode="lines", name="Equity", line=dict(width=2),
    ))
    fig.update_layout(
        title="Curva de equity",
        xaxis_title="Fecha de cierre", yaxis_title="Equity (USD)",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_montecarlo_results(resultado_mc: ResultadoMonteCarlo) -> None:
    """Barras: P(Éxito) vs P(Ruina) vs P(Indeterminado), sobre N iteraciones."""
    df = pd.DataFrame({
        "resultado": ["Éxito", "Ruina", "Indeterminado"],
        "probabilidad": [
            resultado_mc.prob_exito,
            resultado_mc.prob_ruina,
            resultado_mc.prob_indeterminado,
        ],
    })

    fig = px.bar(
        df, x="resultado", y="probabilidad", color="resultado",
        color_discrete_map={"Éxito": "#2ecc71", "Ruina": "#e74c3c", "Indeterminado": "#95a5a6"},
        text="probabilidad",
        title=f"Monte Carlo ({resultado_mc.n_simulaciones:,} iteraciones)",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        yaxis_title="Probabilidad (%)", xaxis_title=None,
        showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
