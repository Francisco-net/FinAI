"""Indicadores econômicos do Banco Central (API pública SGS), com fallback offline.

Séries SGS usadas:
  432   - Meta Selic (% a.a.)
  4389  - CDI anualizado base 252 (% a.a.)
  13522 - IPCA acumulado em 12 meses (%)
  1     - Dólar PTAX venda (R$/US$)
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel

from finai.config import get_settings
from finai.knowledge.assets import load_assumptions
from finai.market.cache import ttl_cache

log = logging.getLogger(__name__)

SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"
SERIES = {"selic_meta": 432, "cdi_anual": 4389, "ipca_12m": 13522, "dolar_ptax": 1}
PERCENT_SERIES = {"selic_meta", "cdi_anual", "ipca_12m"}


class Indicador(BaseModel):
    valor: float
    data: str
    fonte: str


class Indicadores(BaseModel):
    selic_meta: Indicador
    cdi_anual: Indicador
    ipca_12m: Indicador
    dolar_ptax: Indicador
    juro_real_ex_post: float
    aviso: str | None = None


def _fetch_series(client: httpx.Client, code: int) -> tuple[float, str]:
    resp = client.get(SGS_URL.format(code=code))
    resp.raise_for_status()
    item = resp.json()[-1]
    return float(item["valor"].replace(",", ".")), item["data"]


def _fallback() -> dict[str, Indicador]:
    fb = load_assumptions()["indicadores_fallback"]
    fonte = f"valor de referência offline ({fb['data_referencia']})"
    return {
        k: Indicador(valor=float(fb[k]), data=fb["data_referencia"], fonte=fonte) for k in SERIES
    }


@ttl_cache(lambda: get_settings().market_cache_ttl_seconds)
def get_indicators() -> Indicadores:
    settings = get_settings()
    values = _fallback()
    failures: list[str] = []

    if not settings.offline:
        with httpx.Client(timeout=settings.bcb_timeout_seconds) as client:
            for name, code in SERIES.items():
                try:
                    valor, data = _fetch_series(client, code)
                    if name in PERCENT_SERIES:
                        valor /= 100
                    values[name] = Indicador(
                        valor=valor, data=data, fonte=f"Banco Central (SGS {code})"
                    )
                except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
                    log.warning("Falha ao buscar série SGS %s: %s", code, exc)
                    failures.append(name)
    else:
        failures = list(SERIES)

    aviso = None
    if failures:
        aviso = (
            "Não foi possível consultar o Banco Central agora para: "
            + ", ".join(failures)
            + ". Foram usados valores de referência que podem estar desatualizados."
        )

    cdi, ipca = values["cdi_anual"].valor, values["ipca_12m"].valor
    return Indicadores(
        **values,
        juro_real_ex_post=round((1 + cdi) / (1 + ipca) - 1, 4),
        aviso=aviso,
    )
