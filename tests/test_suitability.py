from finai.domain.models import RespostasSuitability
from finai.domain.suitability import avaliar_perfil

ARROJADO = dict(
    objetivo="maximizar_crescimento",
    horizonte="mais_de_10_anos",
    reacao_queda_20="comprar_mais",
    conhecimento="avancado",
    experiencia_anos="mais_de_5",
    estabilidade_renda="muito_estavel",
    necessidade_liquidez="nenhuma",
)


def test_respostas_agressivas_resultam_em_arrojado():
    res = avaliar_perfil(RespostasSuitability(**ARROJADO))
    assert res.perfil == "arrojado"
    assert res.pontuacao == 100
    assert res.experiencia_sugerida == "avancado"
    assert res.travas_aplicadas == []


def test_respostas_cautelosas_resultam_em_conservador():
    res = avaliar_perfil(
        RespostasSuitability(
            objetivo="preservar_capital",
            horizonte="ate_1_ano",
            reacao_queda_20="vender_tudo",
            conhecimento="nenhum",
            experiencia_anos="nunca",
            estabilidade_renda="instavel",
            necessidade_liquidez="alta",
        )
    )
    assert res.perfil == "conservador"
    assert res.pontuacao == 0
    assert res.experiencia_sugerida == "iniciante"


def test_trava_de_reacao_limita_perfil_mesmo_com_pontuacao_alta():
    res = avaliar_perfil(RespostasSuitability(**{**ARROJADO, "reacao_queda_20": "vender_tudo"}))
    assert res.perfil_pela_pontuacao == "arrojado"
    assert res.perfil == "conservador"
    assert any("Venderia tudo" in t for t in res.travas_aplicadas)


def test_sem_conhecimento_limita_a_moderado():
    res = avaliar_perfil(RespostasSuitability(**{**ARROJADO, "conhecimento": "nenhum"}))
    assert res.perfil == "moderado"
