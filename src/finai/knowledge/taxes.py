"""Regras tributárias da renda fixa para pessoa física (IR regressivo, IOF, poupança)."""

from __future__ import annotations

from finai.knowledge.assets import load_assumptions


def _tax_cfg() -> dict:
    return load_assumptions()["tributacao"]


def ir_rate(days: int) -> float:
    """Alíquota do IR regressivo para um prazo em dias corridos."""
    for faixa in _tax_cfg()["ir_regressivo"]:
        if faixa["ate_dias"] is None or days <= faixa["ate_dias"]:
            return float(faixa["aliquota"])
    raise AssertionError("tabela de IR sem faixa final")


def iof_rate(days: int) -> float:
    """Fração do rendimento cobrada de IOF para resgates antes de 30 dias."""
    table = _tax_cfg()["iof_regressivo"]
    if days <= 0:
        return 1.0
    if days >= 30:
        return 0.0
    return table[days - 1] / 100


def is_tax_exempt(product: str) -> bool:
    return product in set(_tax_cfg()["isentos_pf"])


def poupanca_annual_rate(selic: float, tr_monthly: float = 0.0) -> float:
    """Rendimento anual da poupança dada a Selic meta (regra pós-2012)."""
    cfg = _tax_cfg()["poupanca"]
    if selic > cfg["selic_gatilho"]:
        monthly = cfg["rendimento_mensal_acima_gatilho"] + tr_monthly
    else:
        monthly = (1 + selic * cfg["fracao_selic_abaixo_gatilho"]) ** (1 / 12) - 1 + tr_monthly
    return (1 + monthly) ** 12 - 1


def custody_fee(product: str, amount: float) -> float:
    """Taxa anual de custódia da B3 aplicável ao Tesouro Direto."""
    cfg = _tax_cfg()
    if not product.startswith("tesouro"):
        return 0.0
    if product == "tesouro_selic" and amount <= cfg["tesouro_selic_isento_custodia_ate"]:
        return 0.0
    return float(cfg["custodia_tesouro_anual"])
