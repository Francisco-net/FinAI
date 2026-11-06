"""Projeção de patrimônio: cálculo determinístico e simulação de Monte Carlo.

Tudo em TERMOS REAIS (R$ de hoje): retornos descontados da inflação e aportes
reajustados pela inflação. Assim, metas podem ser expressas em valores atuais.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from finai.domain.models import SimulacaoEntrada
from finai.portfolio.allocation import from_percent, model_portfolio
from finai.portfolio.metrics import portfolio_return, portfolio_volatility, real_return


class PontoProjecao(BaseModel):
    ano: int
    p10: float
    p50: float
    p90: float
    total_aportado: float


class ResultadoSimulacao(BaseModel):
    anos: float
    retorno_real_esperado: float
    volatilidade: float
    valor_final_p10: float
    valor_final_p50: float
    valor_final_p90: float
    valor_final_deterministico: float
    total_aportado: float
    prob_atingir_meta: float | None
    aporte_necessario_para_meta: float | None
    serie_anual: list[PontoProjecao]
    observacao: str = (
        "Valores em R$ de hoje (descontada a inflação). p10 = cenário pessimista, "
        "p50 = mediano, p90 = otimista. Projeções não são garantia de retorno."
    )


def monthly_rate(annual: float) -> float:
    return (1 + annual) ** (1 / 12) - 1


def future_value(pv: float, pmt: float, annual_rate: float, months: int) -> float:
    i = monthly_rate(annual_rate)
    if abs(i) < 1e-12:
        return pv + pmt * months
    growth = (1 + i) ** months
    return pv * growth + pmt * (growth - 1) / i


def required_pmt(target: float, pv: float, annual_rate: float, months: int) -> float:
    """Aporte mensal necessário para sair de ``pv`` e chegar em ``target`` em ``months`` meses."""
    if months <= 0:
        return max(target - pv, 0.0)
    i = monthly_rate(annual_rate)
    if abs(i) < 1e-12:
        return max((target - pv) / months, 0.0)
    growth = (1 + i) ** months
    pmt = (target - pv * growth) * i / (growth - 1)
    return max(pmt, 0.0)


def simulate(
    valor_inicial: float,
    aporte_mensal: float,
    anos: float,
    retorno_nominal: float,
    volatilidade: float,
    meta: float | None = None,
    n_cenarios: int = 5000,
    seed: int | None = 42,
) -> ResultadoSimulacao:
    months = max(1, round(anos * 12))
    r_real = real_return(retorno_nominal)

    # Parâmetros lognormais equivalentes ao retorno aritmético/volatilidade anual
    s2 = math.log(1 + (volatilidade**2) / (1 + r_real) ** 2)
    m = math.log(1 + r_real) - s2 / 2
    mu_m, sigma_m = m / 12, math.sqrt(s2 / 12)

    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu_m, sigma_m, size=(n_cenarios, months))
    growth = np.exp(shocks)

    wealth = np.full(n_cenarios, float(valor_inicial))
    serie: list[PontoProjecao] = []
    for t in range(months):
        wealth = wealth * growth[:, t] + aporte_mensal
        if (t + 1) % 12 == 0 or t == months - 1:
            p10, p50, p90 = np.percentile(wealth, [10, 50, 90])
            serie.append(
                PontoProjecao(
                    ano=math.ceil((t + 1) / 12),
                    p10=round(float(p10), 2),
                    p50=round(float(p50), 2),
                    p90=round(float(p90), 2),
                    total_aportado=round(valor_inicial + aporte_mensal * (t + 1), 2),
                )
            )

    p10, p50, p90 = np.percentile(wealth, [10, 50, 90])
    prob = float(np.mean(wealth >= meta)) if meta else None
    pmt_meta = required_pmt(meta, valor_inicial, r_real, months) if meta else None

    return ResultadoSimulacao(
        anos=anos,
        retorno_real_esperado=round(r_real, 4),
        volatilidade=round(volatilidade, 4),
        valor_final_p10=round(float(p10), 2),
        valor_final_p50=round(float(p50), 2),
        valor_final_p90=round(float(p90), 2),
        valor_final_deterministico=round(
            future_value(valor_inicial, aporte_mensal, r_real, months), 2
        ),
        total_aportado=round(valor_inicial + aporte_mensal * months, 2),
        prob_atingir_meta=round(prob, 4) if prob is not None else None,
        aporte_necessario_para_meta=round(pmt_meta, 2) if pmt_meta is not None else None,
        serie_anual=serie,
    )


def simular_entrada(entrada: SimulacaoEntrada) -> ResultadoSimulacao:
    """Simulação a partir de um perfil (carteira-modelo) ou de uma alocação explícita."""
    if entrada.alocacao:
        weights = from_percent(entrada.alocacao)
    else:
        assert entrada.perfil_risco is not None  # garantido pelo validador do modelo
        weights = model_portfolio(entrada.perfil_risco)
    return simulate(
        valor_inicial=entrada.valor_inicial,
        aporte_mensal=entrada.aporte_mensal,
        anos=entrada.anos,
        retorno_nominal=portfolio_return(weights),
        volatilidade=portfolio_volatility(weights),
        meta=entrada.meta,
    )
