"""Ferramentas que o agente pode chamar.

Cada ferramenta = modelo Pydantic de entrada (vira o JSON Schema enviado ao Claude e
valida a entrada recebida) + função Python determinística. Os números da carteira vêm
SEMPRE daqui, nunca "da cabeça" do modelo de linguagem.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from finai.domain.models import PerfilInvestidor, RespostasSuitability, SimulacaoEntrada, TipoRenda
from finai.domain.suitability import avaliar_perfil
from finai.knowledge.assets import ASSET_CLASSES, asset_class_info
from finai.knowledge.glossary import lookup
from finai.market.bcb import get_indicators
from finai.market.quotes import get_quote
from finai.portfolio.fixed_income import ComparacaoEntrada, comparar
from finai.portfolio.planner import montar_carteira
from finai.portfolio.reserve import calcular_reserva
from finai.portfolio.simulation import simular_entrada


# --------------------------------------------------------------------------- entradas extras
class SemParametros(BaseModel):
    pass


class CotacoesEntrada(BaseModel):
    tickers: list[str] = Field(
        min_length=1, max_length=10, description="Ex.: ['BOVA11', 'IVVB11', 'AAPL']"
    )


class ConceitoEntrada(BaseModel):
    termo: str = Field(description="Conceito a explicar, ex.: 'CDI', 'come-cotas', 'FII'")


class ReservaEntrada(BaseModel):
    despesas_mensais: float = Field(ge=0)
    tipo_renda: TipoRenda
    dependentes: int = Field(default=0, ge=0)
    valor_ja_guardado: float = Field(default=0, ge=0)
    aporte_mensal: float = Field(default=0, ge=0)


# --------------------------------------------------------------------------- handlers
def _comparar(entrada: ComparacaoEntrada) -> Any:
    ind = get_indicators()
    resultados = comparar(
        entrada, cdi=ind.cdi_anual.valor, ipca=ind.ipca_12m.valor, selic=ind.selic_meta.valor
    )
    return {
        "premissas": {
            "cdi_anual": entrada.cdi_anual / 100
            if entrada.cdi_anual is not None
            else ind.cdi_anual.valor,
            "ipca_anual": entrada.ipca_anual / 100
            if entrada.ipca_anual is not None
            else ind.ipca_12m.valor,
            "fonte_indicadores": ind.cdi_anual.fonte,
            "observacao": "Taxas mantidas constantes no prazo; dias corridos/365 (aproximação).",
        },
        "ranking_por_valor_liquido": [r.model_dump() for r in resultados],
    }


def _reserva(entrada: ReservaEntrada) -> Any:
    return calcular_reserva(
        entrada.despesas_mensais,
        entrada.tipo_renda,
        entrada.dependentes,
        entrada.valor_ja_guardado,
        entrada.aporte_mensal,
    )


def _classes(_: SemParametros) -> Any:
    return [info.model_dump() for info in asset_class_info().values()]


# --------------------------------------------------------------------------- registro
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any], Any]

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": json_schema(self.input_model),
            # Com streaming, a entrada da ferramenta chega conforme é gerada; validamos com Pydantic.
            "eager_input_streaming": True,
        }


TOOLS: list[Tool] = [
    Tool(
        "avaliar_perfil_investidor",
        "Calcula o perfil de risco (conservador/moderado/arrojado) a partir do questionário de "
        "suitability. Use quando tiver as respostas do usuário sobre objetivo, horizonte, reação a "
        "quedas, conhecimento, experiência, estabilidade de renda e necessidade de liquidez.",
        RespostasSuitability,
        avaliar_perfil,
    ),
    Tool(
        "montar_carteira",
        "Monta a carteira de investimentos personalizada: reserva de emergência, uma 'caixinha' por "
        "objetivo (alocação conforme prazo e perfil), aporte mensal necessário por objetivo, alocação "
        "consolidada com exemplos de produtos, métricas de risco/retorno, alertas e próximos passos. "
        "Use sempre que for propor ou revisar uma carteira; não invente percentuais sem esta ferramenta.",
        PerfilInvestidor,
        montar_carteira,
    ),
    Tool(
        "simular_patrimonio",
        "Projeta a evolução do patrimônio (Monte Carlo, em R$ de hoje) com aporte mensal, prazo e "
        "perfil ou alocação. Retorna cenários pessimista/mediano/otimista, probabilidade de atingir a "
        "meta e o aporte necessário para ela.",
        SimulacaoEntrada,
        simular_entrada,
    ),
    Tool(
        "comparar_renda_fixa",
        "Compara opções de renda fixa (CDB, LCI/LCA, Tesouro, debêntures, poupança, fundos DI) pela "
        "rentabilidade LÍQUIDA de IR, IOF e custos, para um mesmo valor e prazo. Use para perguntas como "
        "'LCI 90% do CDI ou CDB 110%?'.",
        ComparacaoEntrada,
        _comparar,
    ),
    Tool(
        "consultar_indicadores",
        "Retorna Selic, CDI, IPCA 12 meses e dólar PTAX mais recentes (Banco Central) e o juro real.",
        SemParametros,
        lambda _: get_indicators(),
    ),
    Tool(
        "consultar_cotacoes",
        "Busca preço atual e variação de 12 meses de ações, ETFs, FIIs ou BDRs (ex.: BOVA11, IVVB11, "
        "XPML11) e ativos estrangeiros. Use para dados factuais, não como recomendação de compra.",
        CotacoesEntrada,
        lambda e: [get_quote(t) for t in e.tickers],
    ),
    Tool(
        "explicar_conceito",
        "Busca a definição de um conceito do mercado financeiro no glossário do FinAI (CDI, FGC, "
        "come-cotas, marcação a mercado, ETF, FII etc.).",
        ConceitoEntrada,
        lambda e: lookup(e.termo),
    ),
    Tool(
        "calcular_reserva_emergencia",
        "Calcula o valor ideal da reserva de emergência (meses de despesas conforme tipo de renda e "
        "dependentes), quanto falta e em quantos meses ela fica pronta.",
        ReservaEntrada,
        _reserva,
    ),
    Tool(
        "listar_classes_de_ativos",
        f"Lista as classes de ativos usadas pelo FinAI ({', '.join(ASSET_CLASSES)}) com descrição, "
        "risco, liquidez, retorno esperado de longo prazo e exemplos de produtos.",
        SemParametros,
        _classes,
    ),
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def tool_definitions() -> list[dict[str, Any]]:
    """Definições na ordem fixa (a ordem estável preserva o cache de prompt)."""
    return [t.to_api() for t in TOOLS]


# --------------------------------------------------------------------------- execução
@dataclass
class ToolOutcome:
    data: Any
    content: str
    is_error: bool = False


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def execute_tool(name: str, raw_input: Any) -> ToolOutcome:
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return ToolOutcome(None, f"Ferramenta desconhecida: {name}", is_error=True)
    if not isinstance(raw_input, dict):
        return ToolOutcome(None, "Entrada inválida: esperado um objeto JSON.", is_error=True)
    try:
        args = tool.input_model.model_validate(raw_input)
    except ValidationError as exc:
        erros = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'entrada'}: {e['msg']}" for e in exc.errors()
        )
        return ToolOutcome(None, f"Entrada inválida para {name}: {erros}", is_error=True)
    try:
        data = _jsonable(tool.handler(args))
    except ValueError as exc:
        return ToolOutcome(None, f"Não foi possível executar {name}: {exc}", is_error=True)
    return ToolOutcome(data, json.dumps(data, ensure_ascii=False))


# --------------------------------------------------------------------------- schema
def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema do Pydantic com $refs resolvidos inline e sem 'title' (menos tokens)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any, in_properties: bool = False) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = defs[node["$ref"].split("/")[-1]]
                merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged)
            out = {}
            for k, v in node.items():
                if k == "title" and not in_properties:
                    continue
                out[k] = resolve(v, in_properties=(k == "properties" and not in_properties))
            return out
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    resolved = resolve(schema)
    resolved.setdefault("properties", {})
    return resolved
