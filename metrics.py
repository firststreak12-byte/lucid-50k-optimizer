"""
metrics.py (antes metrics_montecarlo.py)
Simulador estocástico de ruina. Responde la pregunta que un solo backtest
lineal NO puede responder: "¿qué tan sensible es el resultado de la
evaluación al ORDEN en que llegaron las rachas de pérdidas?".

Metodología: bootstrap sampling (reordenamiento aleatorio CON reemplazo)
de la secuencia de retornos por trade — el mismo conjunto de operaciones
que arrojó el backtest, pero viviendo N=10,000 historias alternativas.

GRANULARIDAD POR TRADE (ya no una aproximación): el input de este módulo
es y siempre fue una lista de retornos POR TRADE. Antes se alimentaba a
`risk_engine.update_eod_state()` tratando cada trade bootstrapeado como si
cerrara su propio "día" — una aproximación conservadora al modelo EOD
original. Ahora que risk_engine.py implementa trailing drawdown INTRADÍA
real (Tradovate/Lucid), cada trade se registra vía
`risk_engine.register_intraday_trade()`, que es exactamente la unidad de
tiempo nativa de este simulador: ya no hay descalce entre lo que Monte
Carlo simula y cómo se audita el riesgo en producción.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from config import Lucid50KConfig
from risk_engine import RiskEngine

Fase = Literal["evaluacion", "funded"]


@dataclass
class ResultadoMonteCarlo:
    n_simulaciones: int
    prob_exito: float           # % de iteraciones que alcanzaron el profit target
    prob_ruina: float           # % de iteraciones que tocaron el piso de drawdown EOD
    prob_indeterminado: float   # % que no tocó ni el target ni el piso con los trades disponibles
    promedio_trades_hasta_exito: Optional[float]
    promedio_trades_hasta_ruina: Optional[float]


class MonteCarloAuditor:
    """
    Recibe la lista de retornos NETOS por trade (ya con comisión y
    slippage descontados, tal como los produce backtester.py) y evalúa
    la probabilidad de superar — o de reventar — la cuenta de Lucid bajo
    reordenamientos aleatorios de esa misma secuencia.
    """

    def __init__(self, retornos_por_trade: List[float], config: Lucid50KConfig, fase: Fase = "evaluacion"):
        if len(retornos_por_trade) < 10:
            raise ValueError(
                f"Se requieren al menos 10 trades para que Monte Carlo tenga validez "
                f"estadística; se recibieron {len(retornos_por_trade)}."
            )
        self.retornos = np.array(retornos_por_trade, dtype=float)
        self.config = config
        self.fase = fase

    def ejecutar(self, n: int = 10_000, semilla: Optional[int] = 42) -> ResultadoMonteCarlo:
        rng = np.random.default_rng(semilla)
        n_trades = len(self.retornos)

        exitos = 0
        ruinas = 0
        trades_hasta_exito: List[int] = []
        trades_hasta_ruina: List[int] = []

        for _ in range(n):
            orden = rng.integers(0, n_trades, size=n_trades)  # bootstrap CON reemplazo
            secuencia = self.retornos[orden]

            motor = RiskEngine(self.config, fase=self.fase)
            equity = self.config.capital_inicial
            resultado_iteracion: Optional[str] = None

            for idx, pnl_trade in enumerate(secuencia, start=1):
                equity += pnl_trade
                motor.register_intraday_trade(equity)

                if motor.cuenta_quemada():
                    ruinas += 1
                    trades_hasta_ruina.append(idx)
                    resultado_iteracion = "ruina"
                    break

                if self.fase == "evaluacion" and (equity - self.config.capital_inicial) >= self.config.profit_target_evaluacion:
                    exitos += 1
                    trades_hasta_exito.append(idx)
                    resultado_iteracion = "exito"
                    break
                if self.fase == "funded" and (equity - self.config.capital_inicial) >= self.config.payout_target_funded:
                    exitos += 1
                    trades_hasta_exito.append(idx)
                    resultado_iteracion = "exito"
                    break

            # si resultado_iteracion sigue None: ni tocó el piso ni la meta
            # con los trades disponibles — se cuenta como indeterminado.

        indeterminados = n - exitos - ruinas

        return ResultadoMonteCarlo(
            n_simulaciones=n,
            prob_exito=exitos / n * 100.0,
            prob_ruina=ruinas / n * 100.0,
            prob_indeterminado=indeterminados / n * 100.0,
            promedio_trades_hasta_exito=float(np.mean(trades_hasta_exito)) if trades_hasta_exito else None,
            promedio_trades_hasta_ruina=float(np.mean(trades_hasta_ruina)) if trades_hasta_ruina else None,
        )


# ============================================================
# Huérfanas de main.py (destruido) — consumidas por optimizer.py y
# report.py. No dependen de RiskEngine ni de Lucid50KConfig: son
# agregaciones puras sobre la lista de TradeFuturo del backtester.
# ============================================================

@dataclass
class MetricasBasicas:
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    ev_por_trade: float


def calcular_metricas_basicas(trades: list) -> MetricasBasicas:
    """`trades`: lista de TradeFuturo (backtester.py). Sin trades, regresa
    métricas en cero en vez de dividir por cero — un grid con <10 días
    operativos ya se descarta río arriba en evaluar_combinacion()."""
    total_trades = len(trades)
    if total_trades == 0:
        return MetricasBasicas(total_trades=0, win_rate_pct=0.0, profit_factor=0.0, ev_por_trade=0.0)

    pnls = [t.pnl_neto for t in trades]
    ganancias = [p for p in pnls if p > 0]
    perdidas = [p for p in pnls if p < 0]

    win_rate_pct = len(ganancias) / total_trades * 100.0
    suma_perdidas = abs(sum(perdidas))
    # Sin pérdidas: profit factor no está definido (división por cero).
    # Se reporta como infinito en vez de 0/None para no penalizar en un
    # sort descendente una racha perfecta — pero nunca se usa para dividir.
    profit_factor = (sum(ganancias) / suma_perdidas) if suma_perdidas > 0 else float("inf")
    ev_por_trade = sum(pnls) / total_trades

    return MetricasBasicas(
        total_trades=total_trades,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        ev_por_trade=ev_por_trade,
    )


def _agregar_pnl_por_dia(trades: list) -> list:
    """`trades`: lista de TradeFuturo. Agrupa pnl_neto por fecha_operativa
    y regresa la serie diaria EN ORDEN CRONOLÓGICO — es el input esperado
    por MonteCarloAuditor (que ya trabaja a granularidad de trade/día
    indistintamente) y por RejillaParametros.evaluar_combinacion() para el
    filtro de "mínimo 10 días operativos"."""
    pnl_por_dia: Dict[object, float] = {}
    for t in trades:
        pnl_por_dia[t.fecha_operativa] = pnl_por_dia.get(t.fecha_operativa, 0.0) + t.pnl_neto

    return [pnl_por_dia[fecha] for fecha in sorted(pnl_por_dia.keys())]
