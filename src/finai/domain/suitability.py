"""Análise de perfil do investidor (suitability).

Pontua as respostas (0-100) e aplica "travas": regras que limitam o perfil
independentemente da pontuação (ex.: quem venderia tudo numa queda não pode ser arrojado).
"""

from __future__ import annotations

from finai.domain.models import Experiencia, PerfilRisco, RespostasSuitability, ResultadoSuitability

_PONTOS: dict[str, dict[str, int]] = {
    "objetivo": {
        "preservar_capital": 0,
        "renda_complementar": 1,
        "crescimento_moderado": 2,
        "maximizar_crescimento": 3,
    },
    "horizonte": {
        "ate_1_ano": 0,
        "1_a_3_anos": 1,
        "3_a_5_anos": 2,
        "5_a_10_anos": 3,
        "mais_de_10_anos": 4,
    },
    "reacao_queda_20": {"vender_tudo": 0, "vender_parte": 1, "manter": 3, "comprar_mais": 4},
    "conhecimento": {"nenhum": 0, "basico": 1, "intermediario": 2, "avancado": 3},
    "experiencia_anos": {"nunca": 0, "menos_de_2": 1, "2_a_5": 2, "mais_de_5": 3},
    "estabilidade_renda": {"instavel": 0, "variavel": 1, "estavel": 2, "muito_estavel": 3},
    "necessidade_liquidez": {"alta": 0, "media": 1, "baixa": 2, "nenhuma": 3},
}
_MAX_PONTOS = sum(max(v.values()) for v in _PONTOS.values())
_ORDEM: tuple[PerfilRisco, ...] = ("conservador", "moderado", "arrojado")

DESCRICOES: dict[PerfilRisco, str] = {
    "conservador": (
        "Prioriza segurança e liquidez. Aceita retornos menores para evitar oscilações; "
        "a carteira é majoritariamente renda fixa, com pequena parcela de diversificação."
    ),
    "moderado": (
        "Busca equilíbrio entre segurança e crescimento. Aceita oscilações moderadas em parte "
        "da carteira para obter ganho real maior no médio/longo prazo."
    ),
    "arrojado": (
        "Foca em crescimento de longo prazo e tolera quedas relevantes no caminho. "
        "Maior parcela em ações, fundos imobiliários e exterior."
    ),
}


def _cap(perfil: PerfilRisco, teto: PerfilRisco) -> PerfilRisco:
    return _ORDEM[min(_ORDEM.index(perfil), _ORDEM.index(teto))]


def avaliar_perfil(respostas: RespostasSuitability) -> ResultadoSuitability:
    pontos = sum(_PONTOS[campo][getattr(respostas, campo)] for campo in _PONTOS)
    score = round(100 * pontos / _MAX_PONTOS, 1)

    perfil_score: PerfilRisco
    if score <= 35:
        perfil_score = "conservador"
    elif score <= 65:
        perfil_score = "moderado"
    else:
        perfil_score = "arrojado"

    perfil = perfil_score
    travas: list[str] = []

    def trava(condicao: bool, teto: PerfilRisco, motivo: str) -> None:
        nonlocal perfil
        if condicao and _ORDEM.index(perfil) > _ORDEM.index(teto):
            perfil = _cap(perfil, teto)
            travas.append(f"{motivo} → limitado a {teto}")

    trava(respostas.horizonte == "ate_1_ano", "conservador", "Horizonte de até 1 ano")
    trava(
        respostas.reacao_queda_20 == "vender_tudo",
        "conservador",
        "Venderia tudo em uma queda de 20%",
    )
    trava(
        respostas.objetivo == "preservar_capital",
        "conservador",
        "Objetivo principal é preservar capital",
    )
    trava(
        respostas.conhecimento == "nenhum", "moderado", "Sem conhecimento prévio de investimentos"
    )
    trava(respostas.horizonte == "1_a_3_anos", "moderado", "Horizonte curto (1 a 3 anos)")
    trava(respostas.necessidade_liquidez == "alta", "moderado", "Alta necessidade de liquidez")

    exp_pts = (
        _PONTOS["conhecimento"][respostas.conhecimento]
        + _PONTOS["experiencia_anos"][respostas.experiencia_anos]
    )
    experiencia: Experiencia = (
        "iniciante" if exp_pts <= 2 else "intermediario" if exp_pts <= 4 else "avancado"
    )

    return ResultadoSuitability(
        perfil=perfil,
        pontuacao=score,
        perfil_pela_pontuacao=perfil_score,
        travas_aplicadas=travas,
        experiencia_sugerida=experiencia,
        descricao=DESCRICOES[perfil],
    )
