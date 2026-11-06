"""Planejador por objetivos (goal-based investing): o coração do FinAI.

Fluxo:
  1. Alertas de saúde financeira (dívidas caras, gastos > renda).
  2. Reserva de emergência primeiro (100% pós-fixado com liquidez diária).
  3. Cada objetivo vira uma "caixinha" com alocação pelo PRAZO + perfil, recebe parte do
     patrimônio atual (por prioridade) e o aporte mensal necessário para bater a meta.
  4. O que sobra vai para "Crescimento de patrimônio" (longo prazo, carteira do perfil).
  5. Consolida tudo em uma alocação única com métricas de risco/retorno.
"""

from __future__ import annotations

from finai.domain.models import (
    AporteDestino,
    Bucket,
    Carteira,
    LinhaAlocacao,
    Objetivo,
    PerfilInvestidor,
    TipoBucket,
)
from finai.knowledge.assets import (
    ASSET_CLASSES,
    asset_class_info,
    cdi_assumption,
    inflation_assumption,
    load_assumptions,
)
from finai.portfolio.allocation import (
    Weights,
    apply_adjustments,
    horizon_portfolio,
    model_portfolio,
    normalize,
    to_percent,
)
from finai.portfolio.metrics import compute_metrics, portfolio_return, real_return
from finai.portfolio.optimizer import optimize
from finai.portfolio.reserve import calcular_reserva
from finai.portfolio.simulation import monthly_rate, required_pmt

_PRIORIDADE = {"alta": 0, "media": 1, "baixa": 2}


def _brl(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _tipo_bucket(prazo_anos: float) -> TipoBucket:
    if prazo_anos <= 2:
        return "curto_prazo"
    if prazo_anos <= 5:
        return "medio_prazo"
    return "longo_prazo"


def _excluded_classes(perfil: PerfilInvestidor) -> set[str]:
    excl: set[str] = set()
    if perfil.preferencias.evitar_cripto or perfil.experiencia == "iniciante":
        excl.add("cripto")
    if perfil.preferencias.evitar_exterior:
        excl.add("internacional")
    if perfil.experiencia == "iniciante":
        excl.add("rf_credito")
    return excl


def _base_weights(perfil: PerfilInvestidor, prazo_anos: float | None) -> Weights:
    longo = prazo_anos is None or prazo_anos > 5
    if longo and perfil.metodo == "otimizado":
        return optimize(perfil.perfil_risco, excluir=_excluded_classes(perfil))
    if prazo_anos is None:
        return model_portfolio(perfil.perfil_risco)
    return horizon_portfolio(perfil.perfil_risco, prazo_anos)


def _vehicles(classe: str, prefere_isentos: bool) -> list[str]:
    veiculos = list(asset_class_info()[classe].veiculos)
    if prefere_isentos:
        veiculos.sort(key=lambda v: 0 if "isent" in v.lower() else 1)
    return veiculos


def montar_carteira(perfil: PerfilInvestidor) -> Carteira:
    alertas: list[str] = []
    notas_ajuste: set[str] = set()
    patrimonio = perfil.patrimonio_investido
    aporte = perfil.aporte_mensal

    # ---- 1. Saúde financeira
    if perfil.possui_dividas_caras:
        alertas.append(
            "PRIORIDADE: quite dívidas caras (cartão rotativo, cheque especial) antes de investir. "
            "Os juros delas superam com folga o retorno de qualquer investimento."
        )
    sobra = perfil.renda_mensal - perfil.despesas_mensais
    if sobra < 0:
        alertas.append(
            f"Suas despesas superam a renda em {_brl(-sobra)}/mês. Ajustar o orçamento vem antes de investir."
        )
    elif aporte > sobra + 1:
        alertas.append(
            f"O aporte informado ({_brl(aporte)}) é maior que a sobra mensal estimada ({_brl(sobra)})."
        )
    elif aporte == 0 and sobra > 0:
        alertas.append(
            f"Sobram cerca de {_brl(sobra)}/mês. Investir parte disso todo mês acelera muito seus objetivos."
        )

    # ---- 2. Reserva de emergência
    reserva = calcular_reserva(
        perfil.despesas_mensais, perfil.tipo_renda, perfil.dependentes, patrimonio, aporte
    )
    restante = patrimonio - reserva.valor_alocado
    meses_fase1 = int(reserva.meses_para_completar or 0)
    if reserva.falta > 0 and reserva.meses_para_completar is None:
        alertas.append("Sem aporte mensal não é possível completar a reserva de emergência.")

    buckets: list[Bucket] = [
        Bucket(
            nome="Reserva de emergência",
            tipo="reserva",
            valor_alvo=reserva.valor_alvo,
            valor_alocado=reserva.valor_alocado,
            aporte_mensal=aporte if reserva.falta > 0 else 0.0,
            alocacao={"rf_pos": 100.0},
            status="Completa"
            if reserva.falta <= 0
            else f"Faltam {_brl(reserva.falta)} (~{meses_fase1} meses)",
        )
    ]
    plano: list[AporteDestino] = []
    if reserva.falta > 0 and aporte > 0:
        plano.append(
            AporteDestino(
                destino="Reserva de emergência", valor_mensal=round(aporte, 2), fase="1_reserva"
            )
        )

    # ---- 3. Objetivos (por prioridade e prazo)
    aporte_disponivel = aporte
    objetivos: list[Objetivo] = sorted(
        perfil.objetivos, key=lambda o: (_PRIORIDADE[o.prioridade], o.prazo_anos)
    )
    for obj in objetivos:
        base = _base_weights(perfil, obj.prazo_anos)
        meses = max(1, round(obj.prazo_anos * 12))
        r_real = real_return(portfolio_return(base))
        pv_necessario = obj.valor_alvo / (1 + monthly_rate(r_real)) ** meses
        alocado = min(restante, pv_necessario)
        restante -= alocado

        meses_aporte = meses - meses_fase1
        pv_inicio_fase2 = alocado * (1 + monthly_rate(r_real)) ** min(meses_fase1, meses)
        if meses_aporte <= 0:
            pmt = 0.0 if pv_inicio_fase2 >= obj.valor_alvo else float("inf")
        else:
            pmt = required_pmt(obj.valor_alvo, pv_inicio_fase2, r_real, meses_aporte)
        aporte_obj = min(pmt, aporte_disponivel)
        aporte_disponivel -= aporte_obj

        weights, notas = apply_adjustments(base, perfil, alocado + aporte_obj * 12)
        notas_ajuste.update(notas)

        if pmt == 0:
            status = "Meta já coberta pelo patrimônio alocado"
        elif pmt == float("inf"):
            status = "Prazo menor que o tempo para montar a reserva"
            alertas.append(
                f"'{obj.nome}': o prazo termina antes de a reserva ficar pronta. Considere estender o prazo "
                "ou usar parte do patrimônio/aporte em paralelo."
            )
        elif aporte_obj + 0.01 >= pmt:
            status = "No caminho: o aporte cobre a meta"
        else:
            falta = pmt - aporte_obj
            status = f"Aporte insuficiente: faltam {_brl(falta)}/mês"
            alertas.append(
                f"'{obj.nome}': para chegar a {_brl(obj.valor_alvo)} em {obj.prazo_anos:g} anos seriam necessários "
                f"{_brl(pmt)}/mês. Opções: aumentar o aporte, estender o prazo ou ajustar a meta."
            )

        buckets.append(
            Bucket(
                nome=obj.nome,
                tipo=_tipo_bucket(obj.prazo_anos),
                prazo_anos=obj.prazo_anos,
                valor_alvo=obj.valor_alvo,
                valor_alocado=round(alocado, 2),
                aporte_mensal=round(aporte_obj, 2),
                aporte_necessario=None if pmt == float("inf") else round(pmt, 2),
                alocacao=to_percent(weights),
                status=status,
            )
        )
        if aporte_obj > 0:
            plano.append(
                AporteDestino(
                    destino=obj.nome, valor_mensal=round(aporte_obj, 2), fase="2_objetivos"
                )
            )

    # ---- 4. Crescimento de patrimônio (sobras de patrimônio e de aporte)
    if restante > 0.01 or aporte_disponivel > 0.01 or not objetivos:
        base = _base_weights(perfil, None)
        weights, notas = apply_adjustments(base, perfil, restante + aporte_disponivel * 12)
        notas_ajuste.update(notas)
        buckets.append(
            Bucket(
                nome="Crescimento de patrimônio (longo prazo)",
                tipo="longo_prazo",
                valor_alocado=round(max(restante, 0.0), 2),
                aporte_mensal=round(aporte_disponivel, 2),
                alocacao=to_percent(weights),
                status="Independência financeira / acúmulo de longo prazo",
            )
        )
        if aporte_disponivel > 0.01:
            plano.append(
                AporteDestino(
                    destino="Crescimento de patrimônio (longo prazo)",
                    valor_mensal=round(aporte_disponivel, 2),
                    fase="2_objetivos",
                )
            )

    # ---- 5. Consolidação
    consolidado = _consolidate(buckets, patrimonio)
    metricas = compute_metrics(consolidado)
    total_ref = patrimonio if patrimonio > 0 else 0.0
    alocacao_consolidada = [
        LinhaAlocacao(
            classe=k,
            nome=asset_class_info()[k].nome,
            percentual=round(v * 100, 1),
            valor=round(v * total_ref, 2),
            veiculos=_vehicles(k, perfil.preferencias.prefere_isentos_ir),
        )
        for k, v in sorted(consolidado.items(), key=lambda kv: -kv[1])
        if v > 1e-4
    ]

    fgc_limite = load_assumptions()["fgc"]["limite_por_cpf_por_instituicao"]
    renda_fixa_bancaria = patrimonio * sum(
        consolidado.get(k, 0) for k in ("rf_pos", "rf_pre", "rf_inflacao")
    )
    if renda_fixa_bancaria > fgc_limite:
        alertas.append(
            f"Ao usar CDB/LCI/LCA, distribua entre instituições: o FGC cobre até {_brl(fgc_limite)} por CPF "
            "por instituição (Tesouro Direto não depende do FGC)."
        )
    alertas.extend(sorted(notas_ajuste))

    return Carteira(
        perfil_risco=perfil.perfil_risco,
        metodo=perfil.metodo,
        patrimonio_total=round(patrimonio, 2),
        aporte_mensal_total=round(aporte, 2),
        reserva=reserva,
        buckets=buckets,
        alocacao_consolidada=alocacao_consolidada,
        metricas=metricas,
        plano_aportes=plano,
        alertas=alertas,
        proximos_passos=_next_steps(perfil, consolidado, reserva.falta > 0),
        premissas={
            "versao": load_assumptions()["versao"],
            "cdi_longo_prazo": cdi_assumption(),
            "inflacao_longo_prazo": inflation_assumption(),
            "observacao": "Metas e projeções em R$ de hoje; retornos esperados são cenário-base, não garantia.",
        },
    )


def _consolidate(buckets: list[Bucket], patrimonio: float) -> Weights:
    """Soma as caixinhas ponderando pelo valor alocado (ou pelos aportes, se ainda não há patrimônio)."""
    total: dict[str, float] = dict.fromkeys(ASSET_CLASSES, 0.0)
    use_value = patrimonio > 0
    for b in buckets:
        peso = b.valor_alocado if use_value else b.aporte_mensal
        for k, pct in b.alocacao.items():
            total[k] += peso * pct / 100
    if sum(total.values()) <= 0:
        return normalize({"rf_pos": 1.0})
    return normalize(total)


def _next_steps(
    perfil: PerfilInvestidor, consolidado: Weights, reserva_pendente: bool
) -> list[str]:
    passos: list[str] = []
    if perfil.possui_dividas_caras:
        passos.append(
            "Negocie e quite as dívidas caras (troque por crédito mais barato, se possível)."
        )
    passos.append(
        "Abra conta em uma corretora regulada pela CVM e com taxa zero para Tesouro Direto e ETFs."
    )
    if reserva_pendente:
        passos.append(
            "Direcione os aportes para a reserva de emergência (Tesouro Selic ou CDB de liquidez diária) até completá-la."
        )
    if any(consolidado.get(k, 0) > 0 for k in ("acoes_br", "fii", "internacional", "cripto")):
        passos.append(
            "Entre em renda variável de forma gradual (ex.: dividir a compra em 3 a 6 meses) para reduzir o risco de entrada."
        )
    passos.append("Programe aportes automáticos mensais logo após receber o salário.")
    passos.append("Rebalanceie a cada 6 meses ou quando uma classe desviar mais de 5 p.p. do alvo.")
    passos.append(
        "Revise o plano anualmente ou em mudanças de vida (emprego, filhos, casamento, herança)."
    )
    return passos
