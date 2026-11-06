import json

import pytest

from finai.agent.tools import TOOLS, execute_tool, tool_definitions


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_schemas_validos_e_sem_refs():
    defs = tool_definitions()
    assert len({d["name"] for d in defs}) == len(defs)
    for d in defs:
        schema = d["input_schema"]
        assert schema["type"] == "object"
        assert d["description"]
        assert all("$ref" not in n for n in _walk(schema))
        assert "$defs" not in schema
        json.dumps(d)  # serializável


def test_definicoes_sao_deterministicas():
    # Ordem/conteúdo estáveis são necessários para o cache de prompt
    assert json.dumps(tool_definitions()) == json.dumps(tool_definitions())


def test_entrada_invalida_vira_erro_para_o_modelo():
    out = execute_tool("montar_carteira", {"idade": "trinta"})
    assert out.is_error
    assert "Entrada inválida" in out.content


def test_ferramenta_desconhecida():
    assert execute_tool("nao_existe", {}).is_error


EXEMPLOS = {
    "avaliar_perfil_investidor": dict(
        objetivo="crescimento_moderado",
        horizonte="5_a_10_anos",
        reacao_queda_20="manter",
        conhecimento="intermediario",
        experiencia_anos="2_a_5",
        estabilidade_renda="estavel",
        necessidade_liquidez="baixa",
    ),
    "montar_carteira": dict(
        idade=40,
        renda_mensal=15000,
        despesas_mensais=9000,
        tipo_renda="empresario",
        patrimonio_investido=200000,
        aporte_mensal=3000,
        perfil_risco="moderado",
        objetivos=[{"nome": "Faculdade dos filhos", "valor_alvo": 300000, "prazo_anos": 12}],
    ),
    "simular_patrimonio": dict(
        valor_inicial=10000, aporte_mensal=1000, anos=10, perfil_risco="moderado", meta=250000
    ),
    "comparar_renda_fixa": dict(
        valor=5000,
        prazo_dias=365,
        opcoes=[
            {"produto": "cdb", "indexador": "pos_cdi", "taxa": 105},
            {"produto": "tesouro_ipca", "indexador": "ipca", "taxa": 7},
        ],
    ),
    "consultar_indicadores": {},
    "consultar_cotacoes": {"tickers": ["BOVA11"]},
    "explicar_conceito": {"termo": "come-cotas"},
    "calcular_reserva_emergencia": dict(
        despesas_mensais=4000, tipo_renda="autonomo", dependentes=1
    ),
    "listar_classes_de_ativos": {},
}


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_todas_as_ferramentas_executam_offline(tool):
    out = execute_tool(tool.name, EXEMPLOS[tool.name])
    assert not out.is_error, out.content
    json.loads(out.content)


def test_glossario_encontra_por_sinonimo_e_acento():
    out = execute_tool("explicar_conceito", {"termo": "Fundo Imobiliário"})
    data = json.loads(out.content)
    assert data["encontrado"] and data["termo"] == "fii"


def test_indicadores_offline_informam_aviso():
    data = json.loads(execute_tool("consultar_indicadores", {}).content)
    assert data["aviso"]
