"""Testa o loop do agente com um cliente falso (sem chamar a API real)."""

from types import SimpleNamespace as NS

from finai.agent.engine import FinAIAgent
from finai.config import Settings

PERFIL = dict(
    idade=30,
    renda_mensal=8000,
    despesas_mensais=5000,
    tipo_renda="clt",
    patrimonio_investido=50000,
    aporte_mensal=2000,
    perfil_risco="moderado",
)


def _usage():
    return NS(
        input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )


def _text_events(text):
    return [
        NS(type="content_block_start", content_block=NS(type="text")),
        NS(type="content_block_delta", delta=NS(type="text_delta", text=text)),
    ]


class FakeStream:
    def __init__(self, events, final):
        self.events, self.final = events, final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final


class FakeClient:
    """Imita client.beta.messages.stream(...) devolvendo respostas pré-definidas."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **params):
        self.calls.append({**params, "messages": list(params["messages"])})
        return self.turns.pop(0)


def _agent(turns, **settings):
    return FinAIAgent(client=FakeClient(turns), settings=Settings(**settings))


def test_loop_executa_ferramenta_e_conclui():
    tool_block = NS(type="tool_use", id="toolu_1", name="montar_carteira", input=PERFIL)
    turn1 = FakeStream(
        [NS(type="content_block_start", content_block=NS(type="tool_use", name="montar_carteira"))],
        NS(content=[tool_block], stop_reason="tool_use", usage=_usage()),
    )
    final_text = NS(type="text", text="Aqui está sua carteira.")
    turn2 = FakeStream(
        _text_events("Aqui está sua carteira."),
        NS(content=[final_text], stop_reason="end_turn", usage=_usage()),
    )
    agent = _agent([turn1, turn2])

    events = list(agent.chat("Monte minha carteira"))
    kinds = [e.kind for e in events]

    assert kinds == ["tool_start", "tool_result", "text", "done"]
    assert agent.last_portfolio["perfil_risco"] == "moderado"
    # histórico append-only: user, assistant(tool_use), user(tool_result), assistant(texto)
    roles = [m["role"] for m in agent.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    result = agent.messages[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1"
    assert "is_error" not in result
    # o conteúdo do assistente é devolvido intacto (preserva blocos de thinking)
    assert agent.messages[1]["content"][0] is tool_block
    assert agent.usage.requests == 2


def test_parametros_da_requisicao():
    turn = FakeStream(
        _text_events("Oi"),
        NS(content=[NS(type="text", text="Oi")], stop_reason="end_turn", usage=_usage()),
    )
    agent = _agent([turn], effort="high")
    list(agent.chat("Olá"))
    params = agent.client.calls[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["thinking"]["type"] == "adaptive"
    assert params["output_config"] == {"effort": "high"}
    assert params["fallbacks"] == "default"
    assert params["betas"] == ["server-side-fallback-2026-07-01"]
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "tool_choice" not in params  # escolha forçada de ferramenta não é suportada no Opus 5.5


def test_fallback_pode_ser_desligado():
    turn = FakeStream(
        [], NS(content=[NS(type="text", text="Oi")], stop_reason="end_turn", usage=_usage())
    )
    agent = _agent([turn], refusal_fallback=False)
    list(agent.chat("Olá"))
    assert "fallbacks" not in agent.client.calls[0]


def test_recusa_nao_entra_no_historico():
    turn = FakeStream(
        _text_events("parcial"), NS(content=[], stop_reason="refusal", usage=_usage())
    )
    agent = _agent([turn])
    events = list(agent.chat("pedido fora da política"))
    assert events[-1].kind == "refusal"
    assert [m["role"] for m in agent.messages] == ["user"]


def test_entrada_invalida_volta_como_erro_e_o_modelo_corrige():
    bad = NS(type="tool_use", id="t1", name="montar_carteira", input={"idade": 30})
    good = NS(type="tool_use", id="t2", name="montar_carteira", input=PERFIL)
    turns = [
        FakeStream([], NS(content=[bad], stop_reason="tool_use", usage=_usage())),
        FakeStream([], NS(content=[good], stop_reason="tool_use", usage=_usage())),
        FakeStream(
            [], NS(content=[NS(type="text", text="ok")], stop_reason="end_turn", usage=_usage())
        ),
    ]
    agent = _agent(turns)
    events = list(agent.chat("carteira"))
    results = [e for e in events if e.kind == "tool_result"]
    assert results[0].is_error and not results[1].is_error
    assert agent.messages[2]["content"][0]["is_error"] is True


def test_sem_credenciais_vira_mensagem_amigavel():
    class NoAuthClient(FakeClient):
        def _stream(self, **params):
            raise TypeError('"Could not resolve authentication method. Expected one of api_key..."')

    agent = FinAIAgent(client=NoAuthClient([]), settings=Settings())
    events = list(agent.chat("oi"))
    assert events[-1].kind == "error"
    assert "ANTHROPIC_API_KEY" in events[-1].text


def test_outros_type_errors_nao_sao_mascarados():
    class BuggyClient(FakeClient):
        def _stream(self, **params):
            raise TypeError("bug de programação")

    agent = FinAIAgent(client=BuggyClient([]), settings=Settings())
    import pytest

    with pytest.raises(TypeError):
        list(agent.chat("oi"))
