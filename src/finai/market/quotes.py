"""Cotações de ativos via yfinance (dependência opcional: ``uv sync --extra market``).

Tickers da B3 recebem o sufixo ``.SA`` automaticamente (ex.: BOVA11 -> BOVA11.SA).
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from finai.config import get_settings
from finai.market.cache import ttl_cache

log = logging.getLogger(__name__)

_B3_PATTERN = re.compile(r"^[A-Z]{4}\d{1,2}[A-Z]?$")  # PETR4, BOVA11, TAEE11, KLBN11, BPAC11F...


class Cotacao(BaseModel):
    ticker: str
    preco: float | None = None
    moeda: str | None = None
    variacao_dia: float | None = None
    variacao_12m: float | None = None
    max_12m: float | None = None
    min_12m: float | None = None
    erro: str | None = None


def normalize_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    if "." not in t and _B3_PATTERN.match(t):
        return f"{t}.SA"
    return t


@ttl_cache(lambda: min(get_settings().market_cache_ttl_seconds, 900))
def get_quote(ticker: str) -> Cotacao:
    symbol = normalize_ticker(ticker)
    if get_settings().offline:
        return Cotacao(ticker=symbol, erro="Modo offline: cotações indisponíveis.")
    try:
        import yfinance as yf
    except ImportError:
        return Cotacao(
            ticker=symbol, erro="Instale o extra 'market' (uv sync --extra market) para cotações."
        )

    try:
        hist = yf.Ticker(symbol).history(period="1y", auto_adjust=True)
        if hist.empty:
            return Cotacao(ticker=symbol, erro="Ticker não encontrado ou sem dados.")
        close = hist["Close"]
        last = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else last
        return Cotacao(
            ticker=symbol,
            preco=round(last, 2),
            moeda="BRL" if symbol.endswith(".SA") else None,
            variacao_dia=round(last / prev - 1, 4),
            variacao_12m=round(last / float(close.iloc[0]) - 1, 4),
            max_12m=round(float(close.max()), 2),
            min_12m=round(float(close.min()), 2),
        )
    except Exception as exc:  # yfinance levanta exceções variadas (rede, parsing, rate limit)
        log.warning("Falha ao buscar cotação de %s: %s", symbol, exc)
        return Cotacao(ticker=symbol, erro=f"Falha ao consultar cotação: {type(exc).__name__}")
