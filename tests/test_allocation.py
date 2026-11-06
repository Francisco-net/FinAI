import pytest

from finai.domain.models import PerfilInvestidor, Preferencias
from finai.knowledge.assets import RISKY_CLASSES
from finai.portfolio.allocation import (
    apply_adjustments,
    from_percent,
    horizon_portfolio,
    model_portfolio,
    risky_share,
)


def _perfil(**kw) -> PerfilInvestidor:
    base = dict(
        idade=35,
        renda_mensal=10000,
        despesas_mensais=6000,
        tipo_renda="clt",
        perfil_risco="arrojado",
        experiencia="avancado",
    )
    return PerfilInvestidor(**{**base, **kw})


@pytest.mark.parametrize("perfil", ["conservador", "moderado", "arrojado"])
def test_carteiras_modelo_somam_100(perfil):
    assert sum(model_portfolio(perfil).values()) == pytest.approx(1.0)


def test_risco_cresce_com_o_perfil():
    shares = [risky_share(model_portfolio(p)) for p in ("conservador", "moderado", "arrojado")]
    assert shares == sorted(shares)


def test_prazo_curto_e_100_por_cento_pos_fixado():
    w = horizon_portfolio("arrojado", 0.5)
    assert w["rf_pos"] == pytest.approx(1.0)


def test_prazo_ate_2_anos_nao_tem_renda_variavel():
    assert risky_share(horizon_portfolio("arrojado", 2)) == 0


def test_prazo_medio_e_um_degrau_mais_conservador():
    assert horizon_portfolio("arrojado", 4) == model_portfolio("moderado")


def test_iniciante_sem_cripto_e_sem_credito_privado():
    w, notas = apply_adjustments(
        model_portfolio("arrojado"), _perfil(experiencia="iniciante"), 100_000
    )
    assert w["cripto"] == 0 and w["rf_credito"] == 0
    assert sum(w.values()) == pytest.approx(1.0)
    assert notas


def test_evitar_exterior_zera_internacional():
    perfil = _perfil(preferencias=Preferencias(evitar_exterior=True, evitar_cripto=True))
    w, _ = apply_adjustments(model_portfolio("arrojado"), perfil, 100_000)
    assert w["internacional"] == 0 and w["cripto"] == 0


def test_idade_reduz_renda_variavel():
    jovem, _ = apply_adjustments(model_portfolio("arrojado"), _perfil(idade=30), 100_000)
    idoso, _ = apply_adjustments(model_portfolio("arrojado"), _perfil(idade=72), 100_000)
    assert risky_share(idoso) == pytest.approx(risky_share(jovem) * 0.5)


def test_from_percent_rejeita_classe_desconhecida():
    with pytest.raises(ValueError):
        from_percent({"bitcoin": 100})


def test_classes_de_risco_existem_no_modelo():
    assert set(model_portfolio("moderado")) >= RISKY_CLASSES
