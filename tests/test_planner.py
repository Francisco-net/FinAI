import pytest

from finai.domain.models import Objetivo, PerfilInvestidor
from finai.portfolio.planner import montar_carteira


def _perfil(**kw) -> PerfilInvestidor:
    base = dict(
        idade=30,
        renda_mensal=8000,
        despesas_mensais=5000,
        tipo_renda="clt",
        patrimonio_investido=50_000,
        aporte_mensal=2000,
        perfil_risco="moderado",
        experiencia="intermediario",
    )
    return PerfilInvestidor(**{**base, **kw})


def test_reserva_vem_primeiro():
    c = montar_carteira(_perfil(patrimonio_investido=10_000))
    reserva = c.buckets[0]
    assert reserva.tipo == "reserva"
    assert reserva.valor_alocado == 10_000  # todo o patrimônio vai para a reserva
    assert c.reserva.valor_alvo == 30_000  # CLT: 6 meses de despesas
    assert c.plano_aportes[0].fase == "1_reserva"


def test_alocacao_consolidada_soma_100():
    c = montar_carteira(
        _perfil(objetivos=[Objetivo(nome="Carro", valor_alvo=60_000, prazo_anos=3)])
    )
    assert sum(linha.percentual for linha in c.alocacao_consolidada) == pytest.approx(100, abs=0.5)
    assert sum(linha.valor for linha in c.alocacao_consolidada) == pytest.approx(50_000, rel=0.01)


def test_aportes_nao_excedem_o_disponivel():
    c = montar_carteira(
        _perfil(
            objetivos=[
                Objetivo(nome="Casa", valor_alvo=500_000, prazo_anos=5, prioridade="alta"),
                Objetivo(nome="Viagem", valor_alvo=20_000, prazo_anos=2),
            ]
        )
    )
    fase2 = sum(a.valor_mensal for a in c.plano_aportes if a.fase == "2_objetivos")
    assert fase2 <= 2000 + 0.01
    assert any("Aporte insuficiente" in b.status for b in c.buckets)
    assert any("Casa" in a for a in c.alertas)


def test_dividas_caras_geram_alerta_prioritario():
    c = montar_carteira(_perfil(possui_dividas_caras=True))
    assert c.alertas[0].startswith("PRIORIDADE")
    assert "dívidas" in c.proximos_passos[0]


def test_sem_patrimonio_usa_aportes_para_consolidar():
    c = montar_carteira(_perfil(patrimonio_investido=0))
    assert c.alocacao_consolidada[0].classe == "rf_pos"  # tudo vai para a reserva no início
    assert c.reserva.meses_para_completar == 15


def test_curto_prazo_nao_tem_renda_variavel():
    c = montar_carteira(
        _perfil(
            perfil_risco="arrojado",
            objetivos=[Objetivo(nome="Intercâmbio", valor_alvo=30_000, prazo_anos=1.5)],
        )
    )
    bucket = next(b for b in c.buckets if b.nome == "Intercâmbio")
    assert set(bucket.alocacao) <= {"rf_pos", "rf_pre"}


def test_patrimonio_alto_gera_alerta_fgc():
    c = montar_carteira(_perfil(patrimonio_investido=2_000_000, perfil_risco="conservador"))
    assert any("FGC" in a for a in c.alertas)


def test_metodo_otimizado_funciona():
    c = montar_carteira(_perfil(metodo="otimizado", patrimonio_investido=300_000))
    longo = next(b for b in c.buckets if b.tipo == "longo_prazo")
    assert sum(longo.alocacao.values()) == pytest.approx(100, abs=0.5)
