"""
optimizer.py (fusión de grid_search_optimizer.py + grid_search_optimizer_2.py)

Búsqueda en malla (grid search) sobre los parámetros de signals.py,
evaluando cada combinación con el pipeline completo (backtester +
RiskEngine con trailing drawdown intradía + MonteCarloAuditor) y
reteniendo solo las que pasan el filtro de aprobación:
    P(Éxito) >= 60%  Y  P(Ruina) <= 5%   (sobre N iteraciones de Monte Carlo)

Dos modos de ejecución, mismo contrato de datos (RejillaParametros,
evaluar_combinacion, refinar_finalistas son compartidos):

  - `ejecutar_grid_search()`      : secuencial, en memoria. Útil para
    mallas chicas, debugging o cuando no vale la pena pagar el overhead
    de levantar un pool de procesos.
  - `ejecutar_grid_search_paralelo()`: PRODUCCIÓN. Reparte la malla entre
    todos los núcleos lógicos vía concurrent.futures.ProcessPoolExecutor
    (el DataFrame OHLCV, la config y el contrato se fijan UNA sola vez
    por proceso hijo vía `initializer`, nunca se re-serializan por tarea)
    y hace streaming de cada resultado a CSV apenas termina (con flush +
    fsync a disco), para no perder nada si el proceso muere a mitad de
    una malla grande y para que la memoria del proceso padre nunca crezca
    con el tamaño de la malla.

Nota de costo computacional: cada combinación corre un backtest completo
+ N iteraciones de Monte Carlo. Para exploración inicial, usar
`n_simulaciones_mc` bajo (500-1,000) y solo subir a 10,000 para las
combinaciones finalistas — `refinar_finalistas()` hace exactamente eso.
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import itertools
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from config import Lucid50KConfig
from risk_engine import RiskEngine
from signals import generar_señales
from backtester import ejecutar_backtest_futuros, FuturesContractSpec
from metrics import (
    MonteCarloAuditor,
    calcular_metricas_basicas,
    _agregar_pnl_por_dia,
    obtener_retornos_por_trade,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optimizer")


# ============================================================
# Definición de la malla y evaluación de una combinación
# (compartido entre el modo secuencial y el paralelo)
# ============================================================

@dataclass
class RejillaParametros:
    """Rangos a explorar. Los múltiplos de ATR van en pasos de 0.5 por
    defecto para mantener la malla manejable; ajustar `paso_atr` si se
    quiere más granularidad (a costa de más combinaciones)."""
    sl_atr_min: float = 1.0
    sl_atr_max: float = 2.5
    tp_atr_min: float = 2.0
    tp_atr_max: float = 5.0
    paso_atr: float = 0.5
    ventanas_rapidas: Sequence[int] = (10, 20)
    ventanas_lentas: Sequence[int] = (30, 50)
    rsi_sobrecompra_valores: Sequence[float] = (65.0, 70.0, 75.0)
    rsi_sobreventa_valores: Sequence[float] = (25.0, 30.0, 35.0)

    def combinaciones(self) -> List[Dict]:
        sl_valores = _rango(self.sl_atr_min, self.sl_atr_max, self.paso_atr)
        tp_valores = _rango(self.tp_atr_min, self.tp_atr_max, self.paso_atr)

        combos = []
        for sl, tp, v_rap, v_len, rsi_oc, rsi_ov in itertools.product(
            sl_valores, tp_valores, self.ventanas_rapidas, self.ventanas_lentas,
            self.rsi_sobrecompra_valores, self.rsi_sobreventa_valores,
        ):
            if v_rap >= v_len:
                continue  # combinación inválida, se descarta sin frenar la malla
            if tp <= sl:
                continue  # un TP más chico que el SL no tiene sentido de risk/reward
            combos.append({
                "multiplo_sl_atr": sl, "multiplo_tp_atr": tp,
                "ventana_rapida": v_rap, "ventana_lenta": v_len,
                "rsi_sobrecompra": rsi_oc, "rsi_sobreventa": rsi_ov,
            })
        return combos


def _rango(minimo: float, maximo: float, paso: float) -> List[float]:
    valores = []
    v = minimo
    while v <= maximo + 1e-9:
        valores.append(round(v, 2))
        v += paso
    return valores


def evaluar_combinacion(
    df: pd.DataFrame,
    params_señal: Dict,
    config: Lucid50KConfig,
    contrato: FuturesContractSpec,
    tipo_contrato: str,
    cantidad_contratos: int,
    fase: str,
    n_simulaciones_mc: int,
) -> Dict:
    """Corre backtest + Monte Carlo para UNA combinación de parámetros.
    Captura excepciones y aplica reglas de aprobación de riesgo optimizadas."""
    try:
        señales = generar_señales(df, **params_señal)
        risk_engine = RiskEngine(config, fase=fase)
        resultado_bt = ejecutar_backtest_futuros(
            df, señales, risk_engine, contrato, tipo_contrato, cantidad_contratos,
        )
        metricas = calcular_metricas_basicas(resultado_bt.trades)
        retornos_trades = obtener_retornos_por_trade(resultado_bt.trades)

        if len(retornos_trades) < 5:
            return {**params_señal, "descartado": f"solo {len(retornos_trades)} trades (<5)"}

        auditor = MonteCarloAuditor(retornos_trades, config, fase=fase, min_trades=5)
        mc = auditor.ejecutar(n=n_simulaciones_mc)

        # Criterio de aprobación optimizado para cuentas de fondeo Lucid:
        # Probabilidad de Éxito >= 60%, Probabilidad de Ruina <= 5% y al menos 15 trades.
        es_aprobado = (mc.prob_exito >= 60.0) and (mc.prob_ruina <= 5.0) and (metricas.total_trades >= 15)

        return {
            **params_señal,
            "total_trades": metricas.total_trades,
            "win_rate_pct": metricas.win_rate_pct,
            "profit_factor": metricas.profit_factor,
            "ev_por_trade": metricas.ev_por_trade,
            "prob_exito": mc.prob_exito,
            "prob_ruina": mc.prob_ruina,
            "aprobado": es_aprobado,
            "descartado": None,
        }
    except ValueError as e:
        return {**params_señal, "descartado": str(e)}


def refinar_finalistas(
    df: pd.DataFrame,
    resultados_grid: pd.DataFrame,
    config: Lucid50KConfig,
    contrato: FuturesContractSpec,
    tipo_contrato: str = "micro",
    cantidad_contratos: int = 5,
    fase: str = "evaluacion",
    top_n: int = 10,
    n_simulaciones_mc: int = 10_000,
) -> pd.DataFrame:
    """Re-evalúa las `top_n` combinaciones (por prob_exito) del grid
    inicial con el N completo de Monte Carlo (10,000 por defecto) exigido
    por el filtro de aprobación oficial."""
    columnas_params = ["multiplo_sl_atr", "multiplo_tp_atr", "ventana_rapida",
                        "ventana_lenta", "rsi_sobrecompra", "rsi_sobreventa"]
    candidatos = resultados_grid[resultados_grid["descartado"].isna()].head(top_n)

    filas = []
    for _, fila in candidatos.iterrows():
        params = {c: fila[c] for c in columnas_params}
        filas.append(evaluar_combinacion(
            df, params, config, contrato, tipo_contrato, cantidad_contratos, fase, n_simulaciones_mc,
        ))

    resultados = pd.DataFrame(filas)
    if "prob_exito" in resultados.columns:
        resultados = resultados.sort_values("prob_exito", ascending=False, na_position="last")
    return resultados.reset_index(drop=True)


# ============================================================
# Modo secuencial — mallas chicas / debugging
# ============================================================

def ejecutar_grid_search(
    df: pd.DataFrame,
    rejilla: RejillaParametros,
    config: Lucid50KConfig,
    contrato: FuturesContractSpec,
    tipo_contrato: str = "micro",
    cantidad_contratos: int = 5,
    fase: str = "evaluacion",
    n_simulaciones_mc: int = 1_000,
) -> pd.DataFrame:
    """Corre TODAS las combinaciones de `rejilla` de forma secuencial y
    regresa un DataFrame ordenado por prob_exito descendente."""
    combos = rejilla.combinaciones()
    filas = []
    for params in combos:
        filas.append(evaluar_combinacion(
            df, params, config, contrato, tipo_contrato, cantidad_contratos, fase, n_simulaciones_mc,
        ))

    resultados = pd.DataFrame(filas)
    if "prob_exito" in resultados.columns:
        resultados = resultados.sort_values("prob_exito", ascending=False, na_position="last")
    return resultados.reset_index(drop=True)


# ============================================================
# Modo paralelo — PRODUCCIÓN: ProcessPoolExecutor + streaming a CSV
# ============================================================

_CAMPOS_PARAMS = [
    "multiplo_sl_atr", "multiplo_tp_atr", "ventana_rapida",
    "ventana_lenta", "rsi_sobrecompra", "rsi_sobreventa",
]
_CAMPOS_METRICAS = [
    "total_trades", "win_rate_pct", "profit_factor", "ev_por_trade",
    "prob_exito", "prob_ruina", "aprobado", "descartado",
]
CAMPOS_RESULTADO_GRID = _CAMPOS_PARAMS + _CAMPOS_METRICAS


# Estado de cada proceso hijo
_worker_df = None
_worker_config: Optional[Lucid50KConfig] = None
_worker_contrato: Optional[FuturesContractSpec] = None
_worker_tipo_contrato: Optional[str] = None
_worker_cantidad_contratos: Optional[int] = None
_worker_fase: Optional[str] = None
_worker_n_simulaciones_mc: Optional[int] = None


def _inicializar_worker(df, config, contrato, tipo_contrato, cantidad_contratos, fase, n_simulaciones_mc) -> None:
    """Inicialización única por worker para evitar re-serialización de datos."""
    global _worker_df, _worker_config, _worker_contrato
    global _worker_tipo_contrato, _worker_cantidad_contratos, _worker_fase, _worker_n_simulaciones_mc
    _worker_df = df
    _worker_config = config
    _worker_contrato = contrato
    _worker_tipo_contrato = tipo_contrato
    _worker_cantidad_contratos = cantidad_contratos
    _worker_fase = fase
    _worker_n_simulaciones_mc = n_simulaciones_mc


def _evaluar_combinacion_en_worker(params_señal: Dict) -> Dict:
    """Tarea ejecutada en el pool de procesos."""
    return evaluar_combinacion(
        _worker_df, params_señal, _worker_config, _worker_contrato,
        _worker_tipo_contrato, _worker_cantidad_contratos, _worker_fase, _worker_n_simulaciones_mc,
    )


def ejecutar_grid_search_paralelo(
    df: pd.DataFrame,
    rejilla: RejillaParametros,
    config: Lucid50KConfig,
    contrato: FuturesContractSpec,
    tipo_contrato: str = "micro",
    cantidad_contratos: int = 5,
    fase: str = "evaluacion",
    n_simulaciones_mc: int = 1_000,
    n_procesos: Optional[int] = None,
    ruta_resultados_temp: str = "grid_results_temp.csv",
    on_progreso: Optional["callable"] = None,
) -> pd.DataFrame:
    """Evalúa la malla completa en paralelo con ProcessPoolExecutor y streaming a disco."""
    combos = rejilla.combinaciones()
    if not combos:
        raise ValueError("La rejilla no produjo ninguna combinación válida (revisa los rangos).")

    n_procesos = n_procesos or os.cpu_count() or 1
    logger.info(f"Grid search paralelo: {len(combos)} combinaciones sobre {n_procesos} procesos.")

    t0 = time.time()
    filas_escritas = 0
    aprobadas_en_vivo = 0

    with open(ruta_resultados_temp, "w", newline="", encoding="utf-8") as f_out:
        escritor = csv.DictWriter(f_out, fieldnames=CAMPOS_RESULTADO_GRID, restval="", extrasaction="ignore")
        escritor.writeheader()

        with cf.ProcessPoolExecutor(
            max_workers=n_procesos,
            initializer=_inicializar_worker,
            initargs=(df, config, contrato, tipo_contrato, cantidad_contratos, fase, n_simulaciones_mc),
        ) as executor:
            futuros = [executor.submit(_evaluar_combinacion_en_worker, params) for params in combos]

            for futuro in cf.as_completed(futuros):
                resultado = futuro.result()

                escritor.writerow(resultado)
                f_out.flush()
                os.fsync(f_out.fileno())
                filas_escritas += 1

                if resultado.get("aprobado"):
                    aprobadas_en_vivo += 1
                    logger.info(f"✅ Combinación APROBADA #{aprobadas_en_vivo}: {resultado}")

                if filas_escritas % 50 == 0:
                    logger.info(f"Progreso: {filas_escritas}/{len(combos)} combinaciones evaluadas...")

                if on_progreso is not None:
                    on_progreso(filas_escritas, len(combos), aprobadas_en_vivo)

    duracion = time.time() - t0
    logger.info(
        f"Grid search paralelo completo: {filas_escritas} combinaciones en {duracion:.1f}s "
        f"({filas_escritas / duracion:.1f} combos/s) -> '{ruta_resultados_temp}'. "
        f"Aprobadas: {aprobadas_en_vivo}."
    )

    resultados = pd.read_csv(ruta_resultados_temp)

    if "aprobado" in resultados.columns:
        resultados["aprobado"] = resultados["aprobado"].fillna(False).astype(bool)

    if "prob_exito" in resultados.columns:
        resultados = resultados.sort_values("prob_exito", ascending=False, na_position="last")
    return resultados.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from data_loader import cargar_datos_csv

    if len(sys.argv) < 2:
        print("Uso: python optimizer.py <ruta_csv> [n_procesos]")
        sys.exit(1)

    df = cargar_datos_csv(sys.argv[1])
    n_procesos_cli = int(sys.argv[2]) if len(sys.argv) > 2 else None

    config = Lucid50KConfig()
    contrato = FuturesContractSpec()
    rejilla = RejillaParametros()

    print(f"Combinaciones a evaluar: {len(rejilla.combinaciones())}")
    resultados = ejecutar_grid_search_paralelo(
        df, rejilla, config, contrato, n_simulaciones_mc=500, n_procesos=n_procesos_cli,
    )
    print(resultados.head(15).to_string(index=False))

    aprobados = resultados[resultados.get("aprobado", False) == True]
    print(f"\nCombinaciones aprobadas (P(Éxito)>=60%, P(Ruina)<=5%): {len(aprobados)}")
    if not aprobados.empty:
        finalistas = refinar_finalistas(df, aprobados, config, contrato, n_simulaciones_mc=10_000)
        print(finalistas.to_string(index=False))
