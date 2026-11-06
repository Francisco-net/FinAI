"""Otimização de carteira (média-variância de Markowitz com âncora na carteira-modelo).

Problema resolvido:

    max   μ·w  -  λ·||w - w_modelo||²
    s.a.  σ(w) <= volatilidade_alvo(perfil)
          Σw = 1,  0 <= w_i <= max_i
          Σ w_renda_variavel <= max_renda_variavel(perfil)
          w_rf_pos >= min_rf_pos(perfil)

A âncora (λ) evita soluções de canto típicas do Markowitz puro, que é muito sensível
a pequenos erros nas premissas de retorno.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from finai.domain.models import PerfilRisco
from finai.knowledge.assets import (
    ASSET_CLASSES,
    RISKY_CLASSES,
    covariance_matrix,
    expected_returns,
    load_assumptions,
)
from finai.portfolio.allocation import Weights, model_portfolio, normalize


def optimize(
    perfil: PerfilRisco,
    excluir: set[str] | None = None,
    anchor_strength: float = 0.25,
) -> Weights:
    limites = load_assumptions()["limites_otimizacao"][perfil]
    excluir = excluir or set()
    mu = expected_returns()
    cov = covariance_matrix()

    anchor_dict = model_portfolio(perfil)
    for cls in excluir:
        anchor_dict[cls] = 0.0
    anchor = np.array([normalize(anchor_dict)[k] for k in ASSET_CLASSES])

    max_por_classe = limites.get("max_por_classe", {})
    bounds = []
    for k in ASSET_CLASSES:
        upper = 0.0 if k in excluir else float(max_por_classe.get(k, 1.0))
        bounds.append((0.0, upper))

    risky_idx = [i for i, k in enumerate(ASSET_CLASSES) if k in RISKY_CLASSES]
    pos_idx = ASSET_CLASSES.index("rf_pos")
    vol_target = float(limites["volatilidade_alvo"])

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: vol_target**2 - w @ cov @ w},
        {"type": "ineq", "fun": lambda w: limites["max_renda_variavel"] - np.sum(w[risky_idx])},
        {"type": "ineq", "fun": lambda w: w[pos_idx] - limites["min_rf_pos"]},
    ]

    def objective(w: np.ndarray) -> float:
        return float(-(mu @ w) + anchor_strength * np.sum((w - anchor) ** 2))

    x0 = np.clip(anchor, [b[0] for b in bounds], [b[1] for b in bounds])
    x0 = x0 / x0.sum()
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        # Fallback seguro: carteira-modelo sem as classes excluídas
        return normalize(anchor_dict)

    w = np.where(result.x < 1e-4, 0.0, result.x)
    return normalize(dict(zip(ASSET_CLASSES, w.tolist(), strict=True)))
