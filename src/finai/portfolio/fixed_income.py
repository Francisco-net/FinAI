"""Comparador de renda fixa: rentabilidade LÍQUIDA (IR, IOF, custódia, isenções).

Responde perguntas como "LCI de 90% do CDI ou CDB de 110%?" e
"Tesouro IPCA+ 7% ou prefixado de 13%?" para um mesmo valor e prazo.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from finai.knowledge import taxes

Produto = Literal[
    "cdb",
    "lci",
    "lca",
    "cri",
    "cra",
    "debenture",
    "debenture_incentivada",
    "tesouro_selic",
    "tesouro_prefixado",
    "tesouro_ipca",
    "poupanca",
    "fundo_di",
]
Indexador = Literal["pos_cdi", "pre", "ipca", "poupanca"]


class OpcaoRendaFixa(BaseModel):
    produto: Produto
    indexador: Indexador = Field(
        description="pos_cdi: taxa = % do CDI (ex.: 110); pre: taxa = % a.a. (ex.: 13.5); "
        "ipca: taxa = juro real % a.a. acima do IPCA (ex.: 7); poupanca: taxa ignorada"
    )
    taxa: float = Field(default=0, ge=0, le=500)
    taxa_administracao: float = Field(default=0, ge=0, le=5, description="% a.a. (fundos)")
    nome: str | None = None


class ComparacaoEntrada(BaseModel):
    valor: float = Field(gt=0, description="Valor investido em R$")
    prazo_dias: int = Field(gt=0, le=365 * 40, description="Prazo em dias corridos")
    opcoes: list[OpcaoRendaFixa] = Field(min_length=1, max_length=10)
    cdi_anual: float | None = Field(default=None, description="CDI % a.a. a usar (padrão: atual)")
    ipca_anual: float | None = Field(
        default=None, description="IPCA % a.a. projetado (padrão: últimos 12 meses)"
    )
    selic_anual: float | None = Field(
        default=None, description="Selic % a.a. (para poupança; padrão: atual)"
    )


class ResultadoOpcao(BaseModel):
    nome: str
    produto: str
    isento_ir: bool
    taxa_bruta_anual: float
    valor_bruto_final: float
    iof: float
    imposto_renda: float
    custos: float
    valor_liquido_final: float
    rentabilidade_liquida_periodo: float
    rentabilidade_liquida_anual: float
    equivalente_percentual_cdi_liquido: float
    observacao: str = ""


def _gross_annual_rate(op: OpcaoRendaFixa, cdi: float, ipca: float, selic: float) -> float:
    if op.indexador == "poupanca" or op.produto == "poupanca":
        return taxes.poupanca_annual_rate(selic)
    if op.indexador == "pos_cdi":
        # % do CDI incide sobre a taxa diária; aproximação anual composta (252 dias úteis)
        daily = (1 + cdi) ** (1 / 252) - 1
        return (1 + daily * op.taxa / 100) ** 252 - 1
    if op.indexador == "pre":
        return op.taxa / 100
    return (1 + ipca) * (1 + op.taxa / 100) - 1


def comparar(
    entrada: ComparacaoEntrada, cdi: float, ipca: float, selic: float
) -> list[ResultadoOpcao]:
    cdi = entrada.cdi_anual / 100 if entrada.cdi_anual is not None else cdi
    ipca = entrada.ipca_anual / 100 if entrada.ipca_anual is not None else ipca
    selic = entrada.selic_anual / 100 if entrada.selic_anual is not None else selic
    anos = entrada.prazo_dias / 365
    resultados: list[ResultadoOpcao] = []

    # referência: CDI líquido de IR (CDB 100% CDI) no mesmo prazo
    cdi_bruto = entrada.valor * ((1 + cdi) ** anos - 1)
    cdi_liquido = cdi_bruto * (1 - taxes.ir_rate(entrada.prazo_dias))

    for op in entrada.opcoes:
        taxa_bruta = _gross_annual_rate(op, cdi, ipca, selic)
        bruto = entrada.valor * (1 + taxa_bruta) ** anos
        custo_anual = taxes.custody_fee(op.produto, entrada.valor) + op.taxa_administracao / 100
        custos = bruto - bruto * (1 - custo_anual) ** anos
        ganho = bruto - custos - entrada.valor

        isento = taxes.is_tax_exempt(op.produto)
        iof = 0.0 if isento else max(ganho, 0) * taxes.iof_rate(entrada.prazo_dias)
        ir = 0.0 if isento else max(ganho - iof, 0) * taxes.ir_rate(entrada.prazo_dias)
        liquido = entrada.valor + ganho - iof - ir

        rent_periodo = liquido / entrada.valor - 1
        rent_anual = (1 + rent_periodo) ** (1 / anos) - 1 if anos > 0 else rent_periodo
        obs = []
        if op.produto in {"cri", "cra", "debenture", "debenture_incentivada"}:
            obs.append("sem FGC: risco de crédito do emissor")
        if op.produto == "fundo_di":
            obs.append("come-cotas não modelado em detalhe")
        if op.produto in {"lci", "lca"}:
            obs.append("verifique a carência mínima")

        resultados.append(
            ResultadoOpcao(
                nome=op.nome or f"{op.produto} ({op.indexador} {op.taxa:g})",
                produto=op.produto,
                isento_ir=isento,
                taxa_bruta_anual=round(taxa_bruta, 5),
                valor_bruto_final=round(bruto, 2),
                iof=round(iof, 2),
                imposto_renda=round(ir, 2),
                custos=round(custos, 2),
                valor_liquido_final=round(liquido, 2),
                rentabilidade_liquida_periodo=round(rent_periodo, 5),
                rentabilidade_liquida_anual=round(rent_anual, 5),
                equivalente_percentual_cdi_liquido=round(
                    100 * (liquido - entrada.valor) / cdi_liquido, 1
                )
                if cdi_liquido > 0
                else 0.0,
                observacao="; ".join(obs),
            )
        )
    return sorted(resultados, key=lambda r: -r.valor_liquido_final)
