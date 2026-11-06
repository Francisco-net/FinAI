"""Alocação estratégica: carteiras-modelo por perfil/horizonte + ajustes pelas características do investidor.

Todas as funções trabalham com ``dict[classe, peso]`` onde os pesos somam 1.0.
"""

from __future__ import annotations

from finai.domain.models import PerfilInvestidor, PerfilRisco
from finai.knowledge.assets import ASSET_CLASSES, RISKY_CLASSES, load_assumptions

Weights = dict[str, float]


def normalize(weights: Weights) -> Weights:
    clean = {k: max(0.0, float(weights.get(k, 0.0))) for k in ASSET_CLASSES}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("Alocação vazia")
    return {k: v / total for k, v in clean.items()}


def from_percent(pcts: dict[str, float]) -> Weights:
    unknown = set(pcts) - set(ASSET_CLASSES)
    if unknown:
        raise ValueError(
            f"Classes desconhecidas: {sorted(unknown)}. Válidas: {list(ASSET_CLASSES)}"
        )
    return normalize({k: v / 100 for k, v in pcts.items()})


def to_percent(weights: Weights, ndigits: int = 1) -> dict[str, float]:
    return {k: round(v * 100, ndigits) for k, v in weights.items() if v > 1e-6}


def model_portfolio(perfil: PerfilRisco) -> Weights:
    return from_percent(load_assumptions()["carteiras_modelo"][perfil])


def horizon_portfolio(perfil: PerfilRisco, prazo_anos: float) -> Weights:
    """Carteira-base conforme o prazo do objetivo: quanto mais curto, mais conservadora.

    - até 1 ano: 100% pós-fixado (liquidez, sem oscilação)
    - até 2 anos: pós + prefixado com vencimento casado
    - 2 a 5 anos: um degrau mais conservador que o perfil
    - acima de 5 anos: carteira-modelo do perfil
    """
    if prazo_anos <= 1:
        return normalize({"rf_pos": 1.0})
    if prazo_anos <= 2:
        return normalize({"rf_pos": 0.7, "rf_pre": 0.3})
    if prazo_anos <= 5:
        if perfil == "arrojado":
            return model_portfolio("moderado")
        if perfil == "moderado":
            return model_portfolio("conservador")
        return normalize({"rf_pos": 0.5, "rf_inflacao": 0.3, "rf_pre": 0.2})
    return model_portfolio(perfil)


def _move(w: Weights, origem: str, destino: str, fracao: float = 1.0) -> None:
    qtd = w.get(origem, 0.0) * fracao
    w[origem] = w.get(origem, 0.0) - qtd
    w[destino] = w.get(destino, 0.0) + qtd


def apply_adjustments(
    weights: Weights, perfil: PerfilInvestidor, patrimonio_bucket: float
) -> tuple[Weights, list[str]]:
    """Ajusta a carteira-base às restrições do investidor. Retorna (pesos, notas explicativas)."""
    w = dict(weights)
    notas: list[str] = []
    pref = perfil.preferencias

    if perfil.experiencia == "iniciante":
        if w.get("rf_credito", 0) > 0:
            _move(w, "rf_credito", "rf_inflacao")
            notas.append(
                "Crédito privado trocado por Tesouro IPCA+ (mais simples para iniciantes)."
            )
        if w.get("cripto", 0) > 0:
            _move(w, "cripto", "rf_pos" if pref.evitar_exterior else "internacional")
            notas.append(
                "Cripto removida: volatilidade extrema não é indicada para quem está começando."
            )

    if pref.evitar_cripto and w.get("cripto", 0) > 0:
        _move(w, "cripto", "acoes_br" if pref.evitar_exterior else "internacional")
        notas.append("Cripto removida por preferência.")

    if pref.evitar_exterior and w.get("internacional", 0) > 0:
        _move(w, "internacional", "acoes_br", 0.5)
        _move(w, "internacional", "rf_inflacao", 1.0)
        notas.append(
            "Exterior removido por preferência (atenção: aumenta a concentração em risco-Brasil)."
        )

    if pref.foco_renda_passiva:
        origem = "internacional" if w.get("internacional", 0) >= 0.05 else "acoes_br"
        if w.get(origem, 0) > 0:
            qtd = min(0.05, w[origem])
            w[origem] -= qtd
            w["fii"] = w.get("fii", 0) + qtd
            notas.append("Mais FIIs pelo foco em renda passiva (rendimentos mensais).")

    # Glide path por idade: reduz renda variável para quem está perto/na aposentadoria
    if perfil.idade >= 60:
        fator = 0.5 if perfil.idade >= 70 else 0.75
        for cls in RISKY_CLASSES:
            _move(w, cls, "rf_inflacao", 1 - fator)
        notas.append(
            f"Renda variável reduzida em {round((1 - fator) * 100)}% pela idade (proteção de patrimônio)."
        )

    # Patrimônio pequeno: menos classes, menos custo e mais simples de executar
    if patrimonio_bucket < 10_000:
        merge = {
            "rf_pre": "rf_pos",
            "rf_credito": "rf_inflacao",
            "multimercado": "rf_pos",
            "cripto": "internacional",
        }
        merged = False
        for origem, destino in merge.items():
            if 0 < w.get(origem, 0) < 0.05:
                _move(w, origem, destino)
                merged = True
        if merged:
            notas.append(
                "Classes pequenas (<5%) agrupadas para simplificar uma carteira em formação."
            )

    return normalize(w), notas


def risky_share(weights: Weights) -> float:
    return sum(v for k, v in weights.items() if k in RISKY_CLASSES)
