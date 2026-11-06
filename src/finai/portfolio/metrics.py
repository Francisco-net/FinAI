"""Métricas de risco e retorno de uma alocação, a partir das premissas de mercado."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from finai.domain.models import Metricas
from finai.knowledge.assets import (
    ASSET_CLASSES,
    cdi_assumption,
    covariance_matrix,
    expected_returns,
    inflation_assumption,
)

Z_95 = 1.6448536269514722


def weights_vector(weights: dict[str, float]) -> np.ndarray:
    return np.array([weights.get(k, 0.0) for k in ASSET_CLASSES])


def portfolio_return(weights: dict[str, float]) -> float:
    return float(weights_vector(weights) @ expected_returns())


def portfolio_volatility(weights: dict[str, float]) -> float:
    w = weights_vector(weights)
    return float(math.sqrt(max(w @ covariance_matrix() @ w, 0.0)))


def real_return(nominal: float, inflation: float | None = None) -> float:
    pi = inflation_assumption() if inflation is None else inflation
    return (1 + nominal) / (1 + pi) - 1


def compute_metrics(weights: dict[str, float]) -> Metricas:
    r = portfolio_return(weights)
    vol = portfolio_volatility(weights)
    cdi = cdi_assumption()
    sharpe = (
        (r - cdi) / vol if vol > 0.01 else None
    )  # Sharpe é instável para carteiras quase sem risco
    var95 = max(0.0, -(r - Z_95 * vol))
    prob_neg = float(norm.cdf(-r / vol)) if vol > 0 else 0.0
    return Metricas(
        retorno_esperado_nominal=round(r, 4),
        retorno_esperado_real=round(real_return(r), 4),
        volatilidade=round(vol, 4),
        sharpe=round(sharpe, 2) if sharpe is not None else None,
        perda_1_ano_cenario_ruim=round(var95, 4),
        prob_retorno_negativo_1_ano=round(prob_neg, 4),
    )
