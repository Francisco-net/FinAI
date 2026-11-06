"""Classes de ativos e premissas de mercado carregadas de ``data/assumptions.yaml``."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel

# Ordem canônica das classes (usada em vetores/matrizes)
ASSET_CLASSES: tuple[str, ...] = (
    "rf_pos",
    "rf_pre",
    "rf_inflacao",
    "rf_credito",
    "multimercado",
    "acoes_br",
    "fii",
    "internacional",
    "cripto",
)

# Classes consideradas "renda variável" para limites de risco
RISKY_CLASSES: frozenset[str] = frozenset({"acoes_br", "fii", "internacional", "cripto"})


class AssetClassInfo(BaseModel):
    chave: str
    nome: str
    descricao: str
    retorno_esperado: float
    volatilidade: float
    risco: int
    liquidez: str
    complexidade: str
    veiculos: list[str]


def _load_yaml(name: str) -> dict[str, Any]:
    text = resources.files("finai.data").joinpath(name).read_text(encoding="utf-8")
    return yaml.safe_load(text)


@lru_cache
def load_assumptions() -> dict[str, Any]:
    return _load_yaml("assumptions.yaml")


@lru_cache
def load_glossary() -> dict[str, Any]:
    return _load_yaml("glossario.yaml")


@lru_cache
def asset_class_info() -> dict[str, AssetClassInfo]:
    raw = load_assumptions()["classes"]
    return {k: AssetClassInfo(chave=k, **raw[k]) for k in ASSET_CLASSES}


def expected_returns() -> np.ndarray:
    info = asset_class_info()
    return np.array([info[k].retorno_esperado for k in ASSET_CLASSES])


def volatilities() -> np.ndarray:
    info = asset_class_info()
    return np.array([info[k].volatilidade for k in ASSET_CLASSES])


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Garante matriz de correlação positiva semidefinida (correlações estimadas à mão podem não ser)."""
    sym = (matrix + matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 1e-8, None)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(psd))
    return psd / np.outer(d, d)


@lru_cache
def _covariance_cached() -> tuple[tuple[float, ...], ...]:
    corr_cfg = load_assumptions()["correlacoes"]
    order = corr_cfg["ordem"]
    if tuple(order) != ASSET_CLASSES:
        raise ValueError("A ordem da matriz de correlação no YAML difere de ASSET_CLASSES")
    corr = _nearest_psd(np.array(corr_cfg["matriz"], dtype=float))
    vol = volatilities()
    cov = np.outer(vol, vol) * corr
    return tuple(tuple(row) for row in cov)


def covariance_matrix() -> np.ndarray:
    return np.array(_covariance_cached())


def inflation_assumption() -> float:
    return float(load_assumptions()["cenario_base"]["inflacao_longo_prazo"])


def cdi_assumption() -> float:
    return float(load_assumptions()["cenario_base"]["cdi_longo_prazo"])
