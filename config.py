"""
config.py
Configuración estructurada e inmutable de las reglas de negocio de Lucid
Trading, cuenta de $50,000 USD. Se usa `frozen=True` a propósito: ningún
módulo río abajo (risk_engine, backtester, metrics_montecarlo) debe poder
mutar estos parámetros en tiempo de ejecución — si algo necesita un valor
distinto, se instancia una config nueva, nunca se parchea la existente.
"""

from __future__ import annotations
from dataclasses import dataclass


# Corte preventivo: el circuit breaker de risk_engine.py se activa al
# consumir este % del límite REAL de la firma (no al tocarlo exactamente),
# para dejar margen de reacción antes de romper la regla real.
SAFETY_BUFFER: float = 0.90


@dataclass(frozen=True)
class Lucid50KConfig:
    # --- Capital y apalancamiento ---
    capital_inicial: float = 50_000.0
    max_contratos_mini: int = 4
    max_contratos_micro: int = 40

    # --- Fase de Evaluación ---
    profit_target_evaluacion: float = 3_000.0
    # Trailing drawdown INTRADÍA (Tradovate/Lucid): el piso se recalcula
    # trade a trade contra el Peak Balance en vivo, no al cierre del día.
    # Se congela en capital_inicial la primera vez que el piso teórico lo
    # alcanza o supera (ver RiskEngine._recalcular_piso).
    max_trailing_drawdown: float = 2_000.0

    # --- Fase Pro Funded ---
    payout_target_funded: float = 500.0
    dll_funded_fijo: float = 1_200.0
    dll_funded_pct_peak: float = 0.60
    # NOTA DE INTERPRETACIÓN: la regla "DLL: $1,200 o 60% del Peak EOD
    # Balance" es ambigua tal como está redactada. Aquí se implementa como
    # dll_funded_actual = min(dll_funded_fijo, dll_funded_pct_peak * peak_equity),
    # es decir, el 60% del balance pico actúa como techo de seguridad
    # (prácticamente nunca vinculante salvo que el balance caiga muy bajo)
    # y $1,200 es, en la práctica, el límite operativo. Si la interpretación
    # real de Lucid es otra (ej. 60% de la GANANCIA acumulada, no del
    # balance total), ajustar `RiskEngine._dll_funded_actual()`.

    # --- Regla de consistencia (Pro Funded) ---
    consistency_pct: float = 0.40  # ningún día puede ser >= 40% del profit total

    # --- Seguridad ---
    safety_buffer: float = SAFETY_BUFFER

    def __post_init__(self):
        if self.capital_inicial <= 0:
            raise ValueError("capital_inicial debe ser positivo")
        if self.max_contratos_mini <= 0 or self.max_contratos_micro <= 0:
            raise ValueError("los límites de contratos deben ser positivos")
        if not (0 < self.safety_buffer <= 1):
            raise ValueError("safety_buffer debe estar en (0, 1]")
        if not (0 < self.consistency_pct < 1):
            raise ValueError("consistency_pct debe estar en (0, 1)")
        if self.max_trailing_drawdown <= 0 or self.dll_funded_fijo <= 0:
            raise ValueError("los límites de drawdown deben ser positivos")
