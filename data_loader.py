"""
data_loader.py
Capa de datos aislada de main.py (destruido). Única responsabilidad: leer
el CSV de OHLCV subido vía st.file_uploader, validarlo y normalizarlo.

REGLA DE MEMORIA (Streamlit Community Cloud): sin @st.cache_data, cada
interacción de un widget en la barra lateral (mover un slider, cambiar la
fase) dispara un rerun completo del script y, con él, un nuevo
pd.read_csv() del archivo completo — sobre datos Tick eso satura la RAM
del servidor en minutos. Con @st.cache_data, Streamlit hashea el
contenido del archivo subido y solo vuelve a parsear si cambia.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

COLUMNAS_REQUERIDAS = ("timestamp", "open", "high", "low", "close")


@st.cache_data(show_spinner="Cargando y validando OHLCV...")
def cargar_datos_csv(archivo_subido) -> pd.DataFrame:
    """
    `archivo_subido`: objeto devuelto por st.file_uploader (o una ruta de
    archivo, para uso fuera de Streamlit vía CLI en optimizer.py).

    Valida columnas requeridas, convierte `timestamp` a datetime y regresa
    el DataFrame ordenado cronológicamente. Lanza ValueError (no la traga
    silenciosamente) si falta alguna columna — un CSV mal formado no debe
    llegar disfrazado de OHLCV válido al backtester.
    """
    df = pd.read_csv(archivo_subido)

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"El CSV no contiene las columnas requeridas: {faltantes}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["open", "high", "low", "close"]].isna().any().any():
        filas_invalidas = df[["open", "high", "low", "close"]].isna().any(axis=1).sum()
        raise ValueError(f"{filas_invalidas} fila(s) con OHLC no numérico tras la conversión.")

    return df
