"""
signals.py
Lógica de entrada/salida, completamente desacoplada de la gestión de
riesgo: este módulo NUNCA importa risk_engine ni config — solo sabe de
precios. Es responsabilidad de backtester.py (o de un futuro puente en
vivo) consultar al RiskEngine antes de actuar sobre cualquier señal que
salga de aquí.

Todos los indicadores están vectorizados con pandas/numpy — nada de
bucles fila a fila.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Accion(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SenalOperativa:
    accion: Accion
    precio_referencia: float
    stop_loss: Optional[float]
    take_profit: Optional[float]


# ============================================================
# Indicadores vectorizados
# ============================================================

def sma(serie: pd.Series, ventana: int) -> pd.Series:
    return serie.rolling(window=ventana, min_periods=ventana).mean()


def ema(serie: pd.Series, ventana: int) -> pd.Series:
    return serie.ewm(span=ventana, adjust=False, min_periods=ventana).mean()


def rsi(serie: pd.Series, ventana: int = 14) -> pd.Series:
    delta = serie.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    avg_ganancia = ganancia.ewm(alpha=1 / ventana, min_periods=ventana, adjust=False).mean()
    avg_perdida = perdida.ewm(alpha=1 / ventana, min_periods=ventana, adjust=False).mean()
    rs = avg_ganancia / avg_perdida.replace(0, np.nan)
    valor = 100 - (100 / (1 + rs))
    return valor.fillna(50)


def bandas_bollinger(serie: pd.Series, ventana: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    media = sma(serie, ventana)
    desviacion = serie.rolling(window=ventana, min_periods=ventana).std()
    return pd.DataFrame({
        "bb_media": media,
        "bb_superior": media + num_std * desviacion,
        "bb_inferior": media - num_std * desviacion,
    })


def macd(serie: pd.Series, rapida: int = 12, lenta: int = 26, señal: int = 9) -> pd.DataFrame:
    ema_rapida = ema(serie, rapida)
    ema_lenta = ema(serie, lenta)
    linea_macd = ema_rapida - ema_lenta
    linea_señal = linea_macd.ewm(span=señal, adjust=False, min_periods=señal).mean()
    histograma = linea_macd - linea_señal
    return pd.DataFrame({"macd": linea_macd, "macd_señal": linea_señal, "macd_hist": histograma})


def cruce_medias(serie: pd.Series, ventana_rapida: int = 10, ventana_lenta: int = 30) -> pd.DataFrame:
    sma_rapida = sma(serie, ventana_rapida)
    sma_lenta = sma(serie, ventana_lenta)
    cruce_arriba = (sma_rapida > sma_lenta) & (sma_rapida.shift(1) <= sma_lenta.shift(1))
    cruce_abajo = (sma_rapida < sma_lenta) & (sma_rapida.shift(1) >= sma_lenta.shift(1))
    return pd.DataFrame({"sma_rapida": sma_rapida, "sma_lenta": sma_lenta,
                          "cruce_arriba": cruce_arriba, "cruce_abajo": cruce_abajo})


def atr(df: pd.DataFrame, ventana: int = 14) -> pd.Series:
    high, low, close_prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        high - low, (high - close_prev).abs(), (low - close_prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / ventana, min_periods=ventana, adjust=False).mean()


# ============================================================
# Generación de señales uniformes (BUY / SELL / HOLD + SL/TP dinámico)
# ============================================================

def generar_señales(
    df: pd.DataFrame,
    ventana_rsi: int = 14,
    rsi_sobrecompra: float = 70.0,
    rsi_sobreventa: float = 30.0,
    ventana_rapida: int = 10,
    ventana_lenta: int = 30,
    ventana_atr: int = 14,
    multiplo_sl_atr: float = 1.5,
    multiplo_tp_atr: float = 3.0,
) -> pd.DataFrame:
    """
    Combina cruce de medias (dirección de tendencia) + RSI (filtro de
    sobrecompra/sobreventa) para producir una señal uniforme por barra.
    SL/TP se calculan dinámicamente como múltiplos del ATR vigente en esa
    barra — no son porcentajes fijos, para adaptarse a la volatilidad real
    del contrato de futuros en cada momento.

    Regresa un DataFrame indexado igual que `df`, con columnas:
    accion (str en {"BUY","SELL","HOLD"}), stop_loss, take_profit.

    Anti look-ahead: todos los indicadores usados son estrictamente hacia
    atrás (rolling/ewm); es responsabilidad del caller (backtester.py)
    ejecutar la señal generada en la barra t en la apertura de la barra
    t+1, nunca en su propio cierre.
    """
    columnas_requeridas = {"open", "high", "low", "close"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"df no contiene las columnas requeridas: {faltantes}")

    rsi_valores = rsi(df["close"], ventana_rsi)
    cruces = cruce_medias(df["close"], ventana_rapida, ventana_lenta)
    atr_valores = atr(df, ventana_atr)

    señal_compra = cruces["cruce_arriba"] & (rsi_valores < rsi_sobrecompra)
    señal_venta = cruces["cruce_abajo"] & (rsi_valores > rsi_sobreventa)

    accion = pd.Series(Accion.HOLD.value, index=df.index, dtype=object)
    accion[señal_compra] = Accion.BUY.value
    accion[señal_venta] = Accion.SELL.value

    stop_loss = pd.Series(np.nan, index=df.index)
    take_profit = pd.Series(np.nan, index=df.index)

    dist_sl = multiplo_sl_atr * atr_valores
    dist_tp = multiplo_tp_atr * atr_valores

    es_compra = accion == Accion.BUY.value
    es_venta = accion == Accion.SELL.value

    stop_loss[es_compra] = df["close"][es_compra] - dist_sl[es_compra]
    take_profit[es_compra] = df["close"][es_compra] + dist_tp[es_compra]
    stop_loss[es_venta] = df["close"][es_venta] + dist_sl[es_venta]
    take_profit[es_venta] = df["close"][es_venta] - dist_tp[es_venta]

    return pd.DataFrame({
        "accion": accion,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    })
