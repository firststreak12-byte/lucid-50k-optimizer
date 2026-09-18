"""
backtester.py
Motor de eventos: recorre la serie temporal barra por barra, ejecuta la
estrategia de signals.py y consulta OBLIGATORIAMENTE a risk_engine.py
antes de registrar cualquier operación. Nunca deja pasar una orden que el
RiskEngine rechazaría en producción — el objetivo de este backtester es
justamente simular la cuenta REAL, restricciones incluidas, no un backtest
"libre" que luego resulte inviable contra las reglas de Lucid.

Fricción modelada:
  - Comisión round-trip por contrato (se cobra una sola vez al cierre,
    representando ida y vuelta).
  - Slippage de 1 a 2 ticks por transacción (se aplica en la ENTRADA y en
    la SALIDA por separado, cada una es una "transacción").

Anti look-ahead: la señal nacida en la barra t se ejecuta en la apertura
de la barra t+1 (vía .shift(1) sobre la columna 'accion' de signals.py).

Riesgo intradía: cada cierre de trade se registra contra risk_engine vía
register_intraday_trade(), que recalcula Peak Balance y trailing drawdown
en el acto (modelo Tradovate/Lucid real), no al cierre del día.
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from risk_engine import RiskEngine
from signals import Accion

TipoContrato = Literal["mini", "micro"]


@dataclass(frozen=True)
class FuturesContractSpec:
    """Especificación monetaria de un contrato de futuros. Los valores por
    defecto corresponden a un ejemplo tipo micro E-mini (MES); ajústalos al
    contrato real que se esté auditando."""
    simbolo: str = "MES"
    tick_size: float = 0.25
    tick_value: float = 1.25          # USD por tick, por contrato
    comision_round_trip: float = 1.24  # USD por contrato, ida y vuelta


@dataclass
class TradeFuturo:
    fecha_entrada: pd.Timestamp
    fecha_salida: pd.Timestamp
    direccion: str            # "BUY" o "SELL"
    precio_entrada: float
    precio_salida: float
    contratos: int
    pnl_bruto: float
    comision: float
    slippage_costo: float
    pnl_neto: float
    motivo_salida: str
    fecha_operativa: object
    equity_tras_cierre: float
    bloqueado_por_riesgo: bool = False  # True si el trade se hubiera abierto pero risk_engine lo bloqueó


@dataclass
class ResultadoBacktest:
    trades: List[TradeFuturo] = field(default_factory=list)
    cuenta_rota: bool = False
    fecha_ruina: Optional[object] = None
    profit_target_alcanzado: bool = False
    fecha_target: Optional[object] = None
    ordenes_bloqueadas_por_riesgo: int = 0


def _costo_slippage_en_precio(contrato: FuturesContractSpec, slippage_ticks: float) -> float:
    return slippage_ticks * contrato.tick_size


def ejecutar_backtest_futuros(
    df: pd.DataFrame,
    señales: pd.DataFrame,
    risk_engine: RiskEngine,
    contrato: FuturesContractSpec,
    tipo_contrato: TipoContrato,
    cantidad_contratos: int,
    slippage_ticks: float = 1.5,
) -> ResultadoBacktest:
    """
    df: OHLCV con columnas timestamp/open/high/low/close (y opcionalmente
        fecha_operativa; se calcula si falta).
    señales: salida de signals.generar_señales(df) — MISMO índice que df.
    risk_engine: instancia YA construida (fase evaluación o funded).
    """
    if "fecha_operativa" not in df.columns:
        df = df.copy()
        df["fecha_operativa"] = pd.to_datetime(df["timestamp"]).dt.date

    if len(df) != len(señales):
        raise ValueError("df y señales deben tener la misma longitud/índice.")

    acciones_ejecutables = señales["accion"].shift(1)     # anti look-ahead
    sl_ejecutables = señales["stop_loss"].shift(1)
    tp_ejecutables = señales["take_profit"].shift(1)

    resultado = ResultadoBacktest()
    equity_running = risk_engine.config.capital_inicial
    daily_profits_history: List[float] = []
    pnl_dia_actual = 0.0
    fecha_dia_actual = df.iloc[0]["fecha_operativa"] if len(df) else None

    en_posicion = False
    direccion_actual: Optional[str] = None
    precio_entrada_puro: Optional[float] = None
    precio_entrada_ejecutado: Optional[float] = None
    sl_actual: Optional[float] = None
    tp_actual: Optional[float] = None
    idx_entrada: Optional[int] = None

    for i in range(1, len(df)):
        fila = df.iloc[i]

        # --- Corte de día: snapshot diario (consistencia/logging) + red de
        # seguridad. La ruina YA se detecta trade a trade (ver bloque de
        # cierre de posición más abajo, vía register_intraday_trade); este
        # chequeo aquí solo cubriría un caso degenerado (equity sin
        # trades entre cortes) y en la práctica es un no-op idempotente. ---
        if fila["fecha_operativa"] != fecha_dia_actual:
            risk_engine.update_eod_state(equity_running)
            daily_profits_history.append(pnl_dia_actual)
            pnl_dia_actual = 0.0
            fecha_dia_actual = fila["fecha_operativa"]

            if risk_engine.cuenta_quemada():
                resultado.cuenta_rota = True
                resultado.fecha_ruina = fecha_dia_actual
                break

            if (equity_running - risk_engine.config.capital_inicial) >= risk_engine.config.profit_target_evaluacion \
                    and risk_engine.fase == "evaluacion":
                resultado.profit_target_alcanzado = True
                resultado.fecha_target = fecha_dia_actual
                break

        # --- Gestión de posición abierta ---
        if en_posicion:
            salida_precio = None
            motivo = None

            if direccion_actual == "BUY":
                if fila["low"] <= sl_actual:
                    salida_precio, motivo = sl_actual, "Stop Loss"
                elif fila["high"] >= tp_actual:
                    salida_precio, motivo = tp_actual, "Take Profit"
            else:  # SELL
                if fila["high"] >= sl_actual:
                    salida_precio, motivo = sl_actual, "Stop Loss"
                elif fila["low"] <= tp_actual:
                    salida_precio, motivo = tp_actual, "Take Profit"

            if salida_precio is not None:
                slip = _costo_slippage_en_precio(contrato, slippage_ticks)
                salida_ejecutada = salida_precio - slip if direccion_actual == "BUY" else salida_precio + slip

                mov_bruto_ticks = (salida_precio - precio_entrada_puro) / contrato.tick_size \
                    if direccion_actual == "BUY" else (precio_entrada_puro - salida_precio) / contrato.tick_size
                pnl_bruto = mov_bruto_ticks * contrato.tick_value * cantidad_contratos

                mov_neto_ticks = (salida_ejecutada - precio_entrada_ejecutado) / contrato.tick_size \
                    if direccion_actual == "BUY" else (precio_entrada_ejecutado - salida_ejecutada) / contrato.tick_size
                comision = contrato.comision_round_trip * cantidad_contratos
                pnl_neto = (mov_neto_ticks * contrato.tick_value * cantidad_contratos) - comision
                slippage_costo = pnl_bruto - (mov_neto_ticks * contrato.tick_value * cantidad_contratos)

                equity_running += pnl_neto
                pnl_dia_actual += pnl_neto

                resultado.trades.append(TradeFuturo(
                    fecha_entrada=df.iloc[idx_entrada]["timestamp"],
                    fecha_salida=fila["timestamp"],
                    direccion=direccion_actual,
                    precio_entrada=precio_entrada_ejecutado,
                    precio_salida=salida_ejecutada,
                    contratos=cantidad_contratos,
                    pnl_bruto=pnl_bruto,
                    comision=comision,
                    slippage_costo=slippage_costo,
                    pnl_neto=pnl_neto,
                    motivo_salida=motivo,
                    fecha_operativa=fila["fecha_operativa"],
                    equity_tras_cierre=equity_running,
                ))

                en_posicion = False
                direccion_actual = None

                # --- Trailing drawdown INTRADÍA: se recalcula el Peak
                # Balance y el piso AQUÍ, en el cierre de este trade —
                # exactamente como lo hace Tradovate en producción — en
                # vez de esperar al corte de día. Si esto revienta el
                # piso, la cuenta se considera quemada en el instante
                # mismo del trade, no horas después al cierre del día.
                risk_engine.register_intraday_trade(equity_running)
                if risk_engine.cuenta_quemada():
                    resultado.cuenta_rota = True
                    resultado.fecha_ruina = fila["fecha_operativa"]
                    break

                # Regla de consistencia: si el día ya es demasiado dominante,
                # se bloquean nuevas entradas por el resto del día (fase funded).
                if not risk_engine.check_consistency_rule(daily_profits_history, pnl_dia_actual):
                    risk_engine.circuit_breaker_activo = True  # bloqueo táctico del día en curso

        # --- Búsqueda de nueva entrada ---
        if not en_posicion:
            accion = acciones_ejecutables.iloc[i]
            if accion in (Accion.BUY.value, Accion.SELL.value):
                mini_qty = cantidad_contratos if tipo_contrato == "mini" else 0
                micro_qty = cantidad_contratos if tipo_contrato == "micro" else 0

                orden_valida = risk_engine.validate_order_size(mini_qty, micro_qty)
                bloqueado_por_breaker = risk_engine.circuit_breaker()

                if not orden_valida or bloqueado_por_breaker:
                    resultado.ordenes_bloqueadas_por_riesgo += 1
                else:
                    slip = _costo_slippage_en_precio(contrato, slippage_ticks)
                    precio_entrada_puro = fila["open"]
                    precio_entrada_ejecutado = fila["open"] + slip if accion == Accion.BUY.value else fila["open"] - slip
                    direccion_actual = accion
                    sl_actual = sl_ejecutables.iloc[i]
                    tp_actual = tp_ejecutables.iloc[i]
                    idx_entrada = i
                    en_posicion = True

    return resultado
