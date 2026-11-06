"""Reserva de emergência: primeira etapa de qualquer planejamento."""

from __future__ import annotations

import math

from finai.domain.models import ReservaEmergencia, TipoRenda
from finai.knowledge.assets import load_assumptions

ONDE_INVESTIR_RESERVA = [
    "Tesouro Selic",
    "CDB de liquidez diária pagando >= 100% do CDI (banco coberto pelo FGC)",
    "Conta remunerada/fundo DI de taxa baixa e resgate D+0",
]


def meses_recomendados(tipo_renda: TipoRenda, dependentes: int = 0) -> int:
    cfg = load_assumptions()
    base = int(cfg["reserva_emergencia_meses"][tipo_renda])
    extra = int(cfg["meses_extra_por_dependente"]) * dependentes
    return min(base + extra, int(cfg["max_meses_reserva"]))


def calcular_reserva(
    despesas_mensais: float,
    tipo_renda: TipoRenda,
    dependentes: int = 0,
    patrimonio_disponivel: float = 0.0,
    aporte_mensal: float = 0.0,
) -> ReservaEmergencia:
    meses = meses_recomendados(tipo_renda, dependentes)
    alvo = round(despesas_mensais * meses, 2)
    alocado = round(min(patrimonio_disponivel, alvo), 2)
    falta = round(alvo - alocado, 2)
    if falta <= 0:
        meses_para_completar: float | None = 0.0
    elif aporte_mensal > 0:
        meses_para_completar = float(math.ceil(falta / aporte_mensal))
    else:
        meses_para_completar = None
    return ReservaEmergencia(
        meses_recomendados=meses,
        valor_alvo=alvo,
        valor_alocado=alocado,
        falta=falta,
        meses_para_completar=meses_para_completar,
        onde_investir=ONDE_INVESTIR_RESERVA,
    )
