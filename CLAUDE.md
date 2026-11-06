# FinAI: guia para o Claude Code

Assistente de investimentos (mercado brasileiro): agente Claude + motor de carteiras determinístico em Python.

## Comandos
- `make check`: lint (ruff) + tipos (mypy) + testes. Rode antes de concluir qualquer mudança.
- `uv run pytest tests/test_x.py -k nome`: teste específico.
- `make exemplo`: exercita o motor de carteiras sem IA.

## Arquitetura
- `src/finai/portfolio/planner.py` é o núcleo: reserva → caixinhas por objetivo → crescimento → consolidação.
- Números de mercado ficam em `src/finai/data/assumptions.yaml`, nunca hardcoded no código.
- `src/finai/agent/engine.py`: loop manual de streaming com ferramentas, independente de interface (CLI, API e Streamlit consomem `AgentEvent`).
- Ferramentas: modelo Pydantic de entrada + função em `agent/tools.py`; o JSON Schema é gerado do modelo.

## Convenções
- Nomes de domínio em português (perfil_risco, aporte_mensal...), código e libs em inglês quando natural.
- Pesos de alocação: `dict[classe, float]` somando 1.0 internamente; percentuais (0-100) só na saída.
- Planejamento em termos REAIS (R$ de hoje): retornos descontados da inflação.
- O agente nunca inventa percentuais: toda alocação vem de `montar_carteira`.

## Regras da API do Claude (Opus 5.5) aplicadas no engine
- Histórico append-only: devolver `response.content` intacto (blocos de thinking incluídos); não editar turnos anteriores.
- `tool_choice` forçado (`any`/`tool`) retorna 400: usar `auto` e orientar pelo prompt.
- Thinking é sempre adaptativo; controlar custo/latência por `output_config.effort`.
- System prompt e lista de ferramentas devem ser estáveis (cache de prompt): nada de datas/IDs no prompt.
- Testes do agente usam `FakeClient` (tests/test_engine.py), sem chamadas reais à API.
