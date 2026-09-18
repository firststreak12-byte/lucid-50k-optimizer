"""
risk_engine.py
Middleware de riesgo. Toda orden y todo cierre de trade DEBEN pasar por
aquí antes de que backtester.py (o, a futuro, un puente de ejecución en
vivo) los registre como válidos. El motor no conoce nada de estrategia ni
de precios de mercado: solo conoce reglas de cuenta.

CAMBIO DE MODELO (Tradovate/Lucid real): el Trailing Drawdown NO se
recalcula al cierre del día — se recalcula trade a trade contra el Peak
Balance en vivo. `register_intraday_trade()` es ahora la ÚNICA fuente de
verdad para el piso de drawdown; `update_eod_state()` se conserva solo
como snapshot de cierre de día (para agregación diaria / logging) y
delega en el método intradía, para que nunca existan dos cálculos del
piso compitiendo entre sí.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal

from config import Lucid50KConfig

Fase = Literal["evaluacion", "funded"]


@dataclass
class EstadoCuenta:
    """Snapshot inmutable del estado de la cuenta en un instante dado,
    pensado para loguear/auditar la evolución del piso de drawdown."""
    equity_actual: float
    peak_equity: float
    piso_drawdown_trailing: float
    congelado: bool
    circuit_breaker_activo: bool
    fase: Fase


class RiskEngine:
    """
    Motor de control de riesgo dependiente de Lucid50KConfig.

    Ciclo de vida esperado por trade/día:
      1. validate_order_size() ANTES de enviar cualquier orden.
      2. register_intraday_trade() al CIERRE DE CADA TRADE — así se
         recalcula el Peak Balance y el piso de trailing drawdown con la
         misma granularidad que usa Tradovate en producción.
      3. check_consistency_rule() antes de aceptar un trade en fase funded.
      4. circuit_breaker() consultado por el caller antes de CUALQUIER
         nueva orden, como última línea de defensa.
      5. update_eod_state() al cierre de cada día operativo, únicamente
         para fines de snapshot/logging diario (no mueve el piso por sí
         mismo — delega en register_intraday_trade).
    """

    def __init__(self, config: Lucid50KConfig, fase: Fase = "evaluacion"):
        self.config = config
        self.fase: Fase = fase

        self.equity_actual: float = config.capital_inicial
        self.peak_equity: float = config.capital_inicial
        self.congelado: bool = False
        self.circuit_breaker_activo: bool = False

        if fase == "evaluacion":
            self.piso_drawdown_trailing: float = config.capital_inicial - config.max_trailing_drawdown
        else:
            self.piso_drawdown_trailing = config.capital_inicial - self._dll_funded_actual()

    # ------------------------------------------------------------
    # Validación de tamaño de orden
    # ------------------------------------------------------------
    def validate_order_size(self, mini_qty: int, micro_qty: int) -> bool:
        """True si la orden respeta los límites de contratos de la firma.
        Fail-fast ante cantidades negativas: eso es un bug del caller, no
        una orden a rechazar silenciosamente."""
        if mini_qty < 0 or micro_qty < 0:
            raise ValueError("mini_qty y micro_qty no pueden ser negativos")
        if mini_qty > self.config.max_contratos_mini:
            return False
        if micro_qty > self.config.max_contratos_micro:
            return False
        return True

    # ------------------------------------------------------------
    # Trailing drawdown INTRADÍA — trade a trade (fuente única de verdad)
    # ------------------------------------------------------------
    def register_intraday_trade(self, equity_running: float) -> EstadoCuenta:
        """
        Se llama al CIERRE DE CADA TRADE (no al cierre del día). Recalcula:
          - peak_equity: Peak Balance en vivo, no el peak-del-día-anterior.
          - piso_drawdown_trailing:
              * Evaluación: trailing desde peak_equity, que se CONGELA en
                capital_inicial la primera vez que el piso teórico
                (peak - max_trailing_drawdown) alcanza o supera el capital
                inicial (comportamiento estándar de cuentas con trailing
                drawdown: una vez "a salvo", el piso deja de subir).
              * Funded: el DLL se recalcula contra el peak_equity vigente
                en ese instante (ver _dll_funded_actual).
          - circuit_breaker_activo, contra el piso YA actualizado.
        """
        self.equity_actual = equity_running
        if equity_running > self.peak_equity:
            self.peak_equity = equity_running

        self._recalcular_piso()
        self._verificar_circuit_breaker()
        return self.snapshot()

    def _recalcular_piso(self) -> None:
        if self.fase == "evaluacion":
            piso_teorico = self.peak_equity - self.config.max_trailing_drawdown
            if not self.congelado:
                if piso_teorico >= self.config.capital_inicial:
                    self.piso_drawdown_trailing = self.config.capital_inicial
                    self.congelado = True
                else:
                    self.piso_drawdown_trailing = piso_teorico
            # si ya está congelado, el piso NO vuelve a moverse
        else:
            self.piso_drawdown_trailing = self.peak_equity - self._dll_funded_actual()

    def update_eod_state(self, closed_equity: float) -> EstadoCuenta:
        """Snapshot de cierre de día para agregación diaria (consistencia,
        logging). NO recalcula el piso de forma independiente — delega en
        register_intraday_trade() para que el trailing drawdown intradía
        siga siendo la única fuente de verdad, incluso si se sigue
        llamando este método al final de cada día operativo."""
        return self.register_intraday_trade(closed_equity)

    def _dll_funded_actual(self) -> float:
        """Ver nota de interpretación en config.py: el 60% del peak actúa
        como techo de seguridad sobre el DLL fijo de $1,200."""
        return min(self.config.dll_funded_fijo, self.config.dll_funded_pct_peak * self.peak_equity)

    # ------------------------------------------------------------
    # Regla de consistencia (solo fase funded)
    # ------------------------------------------------------------
    def check_consistency_rule(self, daily_profits_history: List[float], current_trade_pnl: float) -> bool:
        """
        `daily_profits_history`: PnL neto de cada día operativo YA CERRADO
        (no incluye el día en curso).
        `current_trade_pnl`: PnL acumulado del día EN CURSO (incluyendo el
        trade que se está evaluando).

        Regresa True si la operación NO viola la regla de consistencia
        (ningún día puede representar >= 40% del profit total acumulado).
        Si el profit total acumulado es <= 0, la regla no aplica (no hay
        "ganancia total" de la cual un día pueda acaparar un porcentaje).
        """
        if self.fase != "funded":
            return True  # la regla de consistencia solo aplica en Pro Funded

        profit_total = sum(daily_profits_history) + current_trade_pnl
        if profit_total <= 0 or current_trade_pnl <= 0:
            return True

        proporcion_dia = current_trade_pnl / profit_total
        return proporcion_dia < self.config.consistency_pct

    # ------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------
    def _limite_real_vigente(self) -> float:
        return self.config.max_trailing_drawdown if self.fase == "evaluacion" else self._dll_funded_actual()

    def _verificar_circuit_breaker(self) -> None:
        drawdown_consumido = self.peak_equity - self.equity_actual
        limite_real = self._limite_real_vigente()
        if limite_real <= 0:
            self.circuit_breaker_activo = True
            return
        self.circuit_breaker_activo = drawdown_consumido >= self.config.safety_buffer * limite_real

    def circuit_breaker(self) -> bool:
        """True si la cuenta está dentro del SAFETY_BUFFER de romper el
        límite real de trailing drawdown: el caller debe bloquear nuevas
        órdenes."""
        return self.circuit_breaker_activo

    def cuenta_quemada(self) -> bool:
        """True si el equity YA cruzó el piso real (violación consumada,
        no solo el buffer de seguridad)."""
        return self.equity_actual <= self.piso_drawdown_trailing

    def snapshot(self) -> EstadoCuenta:
        return EstadoCuenta(
            equity_actual=self.equity_actual,
            peak_equity=self.peak_equity,
            piso_drawdown_trailing=self.piso_drawdown_trailing,
            congelado=self.congelado,
            circuit_breaker_activo=self.circuit_breaker_activo,
            fase=self.fase,
        )
