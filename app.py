"""
app.py
UI de Streamlit para el pipeline Lucid 50K (config -> risk_engine ->
backtester -> optimizer -> metrics).

PROBLEMA DE ARQUITECTURA QUE ESTO RESUELVE:
Streamlit re-ejecuta TODO el script de arriba a abajo en cada interacción
del usuario (cada click, cada widget). Si el grid search (que puede tardar
minutos y usa un ProcessPoolExecutor con todos los núcleos del CPU) se
llamara directamente en el cuerpo del script, la UI se congelaría por
completo durante toda la corrida, y cualquier click del usuario mientras
tanto relanzaría la búsqueda desde cero.

SOLUCIÓN: el grid search corre en un hilo (`threading.Thread`) DAEMON,
separado del hilo principal de Streamlit. Ese hilo es quien crea y
gestiona el ProcessPoolExecutor (en optimizer.py). El hilo de Streamlit
nunca bloquea: solo lee un estado compartido (`st.session_state` +
un dict de progreso protegido por el GIL) y se refresca solo con
`st.fragment(run_every=...)`, sin tocar la lógica de negocio.

Aislamiento de responsabilidades:
  - optimizer.py       -> SABE ejecutar la malla en paralelo. No sabe que
                           existe Streamlit.
  - app.py (este archivo) -> SABE pintar widgets y leer progreso. No
                           construye DataFrames de resultados ni conoce
                           los detalles de RiskEngine/backtester.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from config import Lucid50KConfig
from backtester import FuturesContractSpec
from optimizer import RejillaParametros, ejecutar_grid_search_paralelo, refinar_finalistas
from data_loader import cargar_datos_csv

RUTA_RESULTADOS_TEMP = "grid_results_temp.csv"


# ============================================================
# Estado compartido entre el hilo de la UI y el hilo del grid search.
# Solo contadores/flags simples (lecturas/escrituras atómicas bajo el
# GIL) — nada de objetos complejos ni llamadas a st.* desde el hilo
# de trabajo: las funciones de Streamlit solo pueden invocarse desde
# el hilo principal del script.
# ============================================================

def _estado_inicial() -> dict:
    return {
        "corriendo": False,
        "filas_escritas": 0,
        "total_combos": 0,
        "aprobadas": 0,
        "resultado_final": None,   # pd.DataFrame una vez terminado
        "error": None,
        "hilo": None,
    }


def _init_session_state() -> None:
    if "grid" not in st.session_state:
        st.session_state.grid = _estado_inicial()
    if "df_ohlcv" not in st.session_state:
        st.session_state.df_ohlcv = None


# ============================================================
# Trabajo en segundo plano — ÚNICA función que toca optimizer.py.
# Corre en un hilo aparte; NO debe llamar a ninguna función de `st`.
# ============================================================

def _lanzar_grid_search_en_hilo(
    df: pd.DataFrame,
    rejilla: RejillaParametros,
    config: Lucid50KConfig,
    contrato: FuturesContractSpec,
    tipo_contrato: str,
    cantidad_contratos: int,
    fase: str,
    n_simulaciones_mc: int,
    n_procesos: Optional[int],
    estado: dict,
) -> None:
    def _callback_progreso(filas_escritas: int, total: int, aprobadas: int) -> None:
        estado["filas_escritas"] = filas_escritas
        estado["total_combos"] = total
        estado["aprobadas"] = aprobadas

    def _target() -> None:
        try:
            resultado = ejecutar_grid_search_paralelo(
                df, rejilla, config, contrato,
                tipo_contrato=tipo_contrato,
                cantidad_contratos=cantidad_contratos,
                fase=fase,
                n_simulaciones_mc=n_simulaciones_mc,
                n_procesos=n_procesos,
                ruta_resultados_temp=RUTA_RESULTADOS_TEMP,
                on_progreso=_callback_progreso,
            )
            estado["resultado_final"] = resultado
        except Exception as exc:  # noqa: BLE001 — se muestra en la UI, no se traga
            estado["error"] = str(exc)
        finally:
            estado["corriendo"] = False

    hilo = threading.Thread(target=_target, daemon=True)
    estado["hilo"] = hilo
    estado["corriendo"] = True
    hilo.start()


# ============================================================
# Panel lateral: parámetros del pipeline
# ============================================================

def _panel_parametros() -> dict:
    st.sidebar.header("Datos")
    archivo = st.sidebar.file_uploader("OHLCV (CSV)", type=["csv"])

    st.sidebar.header("Cuenta Lucid 50K")
    fase = st.sidebar.selectbox("Fase", ["evaluacion", "funded"])
    tipo_contrato = st.sidebar.selectbox("Tipo de contrato", ["micro", "mini"])
    cantidad_contratos = st.sidebar.number_input("Contratos por trade", min_value=1, value=5)

    st.sidebar.header("Rejilla de parámetros")
    col1, col2 = st.sidebar.columns(2)
    sl_min = col1.number_input("SL ATR min", value=1.0, step=0.5)
    sl_max = col2.number_input("SL ATR max", value=2.5, step=0.5)
    tp_min = col1.number_input("TP ATR min", value=2.0, step=0.5)
    tp_max = col2.number_input("TP ATR max", value=5.0, step=0.5)

    st.sidebar.header("Monte Carlo")
    n_simulaciones_mc = st.sidebar.select_slider(
        "Simulaciones por combinación (grid grueso)",
        options=[200, 500, 1_000, 2_000], value=500,
    )
    n_procesos = st.sidebar.number_input(
        "Núcleos a usar (0 = todos)", min_value=0, value=0, step=1,
    )

    return {
        "archivo": archivo,
        "fase": fase,
        "tipo_contrato": tipo_contrato,
        "cantidad_contratos": int(cantidad_contratos),
        "rejilla": RejillaParametros(sl_atr_min=sl_min, sl_atr_max=sl_max, tp_atr_min=tp_min, tp_atr_max=tp_max),
        "n_simulaciones_mc": int(n_simulaciones_mc),
        "n_procesos": int(n_procesos) or None,
    }


# ============================================================
# Fragmento de progreso — se refresca solo, sin recargar toda la app.
# Este es el único punto donde el hilo de la UI "consulta" al hilo de
# trabajo; nunca llama directamente a optimizer.py.
# ============================================================

@st.fragment(run_every=1.5)
def _panel_progreso() -> None:
    estado = st.session_state.grid

    if estado["error"]:
        st.error(f"El grid search falló: {estado['error']}")
        return

    if estado["resultado_final"] is not None:
        st.success("Grid search completo.")
        _mostrar_resultados(estado["resultado_final"])
        return

    if estado["corriendo"]:
        total = estado["total_combos"] or 1
        avance = estado["filas_escritas"] / total if estado["total_combos"] else 0.0
        st.progress(min(avance, 1.0), text=f"{estado['filas_escritas']}/{estado['total_combos']} combinaciones")
        st.caption(f"Aprobadas hasta ahora: {estado['aprobadas']}")

        ruta_temp = Path(RUTA_RESULTADOS_TEMP)
        if ruta_temp.exists():
            try:
                parcial = pd.read_csv(ruta_temp)
                st.caption("Vista parcial (streaming desde disco):")
                st.dataframe(parcial.tail(10), use_container_width=True)
            except pd.errors.EmptyDataError:
                pass


def _mostrar_resultados(resultados: pd.DataFrame) -> None:
    st.dataframe(resultados.head(20), use_container_width=True)
    st.download_button(
        "Descargar resultados completos (CSV)",
        data=resultados.to_csv(index=False).encode("utf-8"),
        file_name="grid_results.csv",
        mime="text/csv",
    )

    aprobados = resultados[resultados.get("aprobado", False) == True]  # noqa: E712
    st.metric("Combinaciones aprobadas", len(aprobados))
    if not aprobados.empty and st.button("Refinar top 10 finalistas (10,000 simulaciones)"):
        with st.spinner("Refinando finalistas..."):
            config = Lucid50KConfig()
            contrato = FuturesContractSpec()
            finalistas = refinar_finalistas(
                st.session_state.df_ohlcv, aprobados, config, contrato,
            )
        st.dataframe(finalistas, use_container_width=True)


# ============================================================
# Main
# ============================================================

def main() -> None:
    st.set_page_config(page_title="Lucid 50K — Optimizador", layout="wide")
    st.title("Lucid 50K — Optimizador de estrategia")
    st.caption(
        "Trailing drawdown intradía (Tradovate/Lucid) + regla de consistencia + "
        "Monte Carlo de ruina, sobre un grid search paralelo con streaming a disco."
    )

    _init_session_state()
    params = _panel_parametros()

    if params["archivo"] is not None and st.session_state.df_ohlcv is None:
        st.session_state.df_ohlcv = cargar_datos_csv(params["archivo"])

    if st.session_state.df_ohlcv is None:
        st.info("Sube un CSV de OHLCV en el panel lateral para empezar.")
        return

    combos = params["rejilla"].combinaciones()
    st.write(f"Combinaciones a evaluar con la rejilla actual: **{len(combos)}**")

    deshabilitado = st.session_state.grid["corriendo"]
    if st.button("Ejecutar grid search", disabled=deshabilitado, type="primary"):
        st.session_state.grid = _estado_inicial()
        _lanzar_grid_search_en_hilo(
            df=st.session_state.df_ohlcv,
            rejilla=params["rejilla"],
            config=Lucid50KConfig(),
            contrato=FuturesContractSpec(),
            tipo_contrato=params["tipo_contrato"],
            cantidad_contratos=params["cantidad_contratos"],
            fase=params["fase"],
            n_simulaciones_mc=params["n_simulaciones_mc"],
            n_procesos=params["n_procesos"],
            estado=st.session_state.grid,
        )
        st.rerun()

    _panel_progreso()


if __name__ == "__main__":
    main()
