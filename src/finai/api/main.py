"""API REST do FinAI (FastAPI).

Rodar:  uv run uvicorn finai.api.main:app --reload
Docs:   http://localhost:8000/docs

Os endpoints determinísticos (/carteira, /perfil, /simulacao...) funcionam sem chave de API.
O /chat usa o agente Claude e responde em Server-Sent Events (SSE).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from finai import __version__
from finai.agent.engine import FinAIAgent
from finai.domain.models import (
    Carteira,
    PerfilInvestidor,
    RespostasSuitability,
    ResultadoSuitability,
    SimulacaoEntrada,
)
from finai.domain.suitability import avaliar_perfil
from finai.market.bcb import Indicadores, get_indicators
from finai.portfolio.fixed_income import ComparacaoEntrada, ResultadoOpcao, comparar
from finai.portfolio.planner import montar_carteira
from finai.portfolio.simulation import ResultadoSimulacao, simular_entrada

app = FastAPI(
    title="FinAI API",
    version=__version__,
    description="Assistente de IA para montagem de carteiras de investimento (mercado brasileiro).",
)

# Sessões de chat em memória (para produção: Redis/banco de dados + autenticação)
_sessions: dict[str, FinAIAgent] = {}
_sessions_lock = threading.Lock()


class ChatRequest(BaseModel):
    mensagem: str = Field(min_length=1, max_length=8000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "versao": __version__}


@app.get("/indicadores", response_model=Indicadores)
def indicadores() -> Indicadores:
    return get_indicators()


@app.post("/perfil", response_model=ResultadoSuitability)
def perfil(respostas: RespostasSuitability) -> ResultadoSuitability:
    return avaliar_perfil(respostas)


@app.post("/carteira", response_model=Carteira)
def carteira(perfil: PerfilInvestidor) -> Carteira:
    return montar_carteira(perfil)


@app.post("/simulacao", response_model=ResultadoSimulacao)
def simulacao(entrada: SimulacaoEntrada) -> ResultadoSimulacao:
    try:
        return simular_entrada(entrada)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/renda-fixa/comparar", response_model=list[ResultadoOpcao])
def comparar_renda_fixa(entrada: ComparacaoEntrada) -> list[ResultadoOpcao]:
    ind = get_indicators()
    return comparar(
        entrada, cdi=ind.cdi_anual.valor, ipca=ind.ipca_12m.valor, selic=ind.selic_meta.valor
    )


def _get_agent(session_id: str) -> FinAIAgent:
    with _sessions_lock:
        if session_id not in _sessions:
            from finai.config import Settings
            _sessions[session_id] = FinAIAgent(settings=Settings(refusal_fallback=False))
        return _sessions[session_id]


@app.post("/chat/{session_id}")
def chat(session_id: str, req: ChatRequest) -> StreamingResponse:
    """Envia uma mensagem e recebe a resposta em streaming (eventos SSE com JSON)."""
    try:
        agent = _get_agent(session_id)
    except Exception as exc:
        raise HTTPException(503, f"Agente indisponível: {exc}") from exc

    def stream() -> Iterator[str]:
        for event in agent.chat(req.mensagem):
            yield f"event: {event.kind}\ndata: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/chat/{session_id}/carteira")
def ultima_carteira(session_id: str) -> dict:
    agent = _sessions.get(session_id)
    if agent is None or agent.last_portfolio is None:
        raise HTTPException(404, "Nenhuma carteira montada nesta sessão ainda.")
    return agent.last_portfolio


@app.delete("/chat/{session_id}")
def encerrar_sessao(session_id: str) -> dict[str, str]:
    with _sessions_lock:
        _sessions.pop(session_id, None)
    return {"status": "encerrada"}
