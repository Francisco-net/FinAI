import pytest

from finai.knowledge.assets import load_assumptions
from finai.portfolio.allocation import risky_share
from finai.portfolio.metrics import portfolio_volatility
from finai.portfolio.optimizer import optimize


@pytest.mark.parametrize("perfil", ["conservador", "moderado", "arrojado"])
def test_otimizador_respeita_limites(perfil):
    limites = load_assumptions()["limites_otimizacao"][perfil]
    w = optimize(perfil)
    assert sum(w.values()) == pytest.approx(1.0)
    assert portfolio_volatility(w) <= limites["volatilidade_alvo"] + 1e-3
    assert risky_share(w) <= limites["max_renda_variavel"] + 1e-3
    assert w["rf_pos"] >= limites["min_rf_pos"] - 1e-3
    for cls, maximo in limites["max_por_classe"].items():
        assert w[cls] <= maximo + 1e-3


def test_otimizador_respeita_exclusoes():
    w = optimize("arrojado", excluir={"cripto", "internacional"})
    assert w["cripto"] == 0 and w["internacional"] == 0
