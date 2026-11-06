import pytest

from finai.portfolio.simulation import future_value, required_pmt, simulate


def test_pmt_e_valor_futuro_sao_inversos():
    pmt = required_pmt(target=100_000, pv=10_000, annual_rate=0.06, months=60)
    assert future_value(10_000, pmt, 0.06, 60) == pytest.approx(100_000, rel=1e-9)


def test_pmt_zero_quando_meta_ja_coberta():
    assert required_pmt(target=1_000, pv=5_000, annual_rate=0.05, months=12) == 0


def test_simulacao_sem_volatilidade_converge_para_deterministico():
    r = simulate(10_000, 500, 10, retorno_nominal=0.10, volatilidade=1e-6, n_cenarios=200)
    assert r.valor_final_p50 == pytest.approx(r.valor_final_deterministico, rel=1e-3)


def test_simulacao_reprodutivel_e_ordenada():
    a = simulate(0, 1000, 20, 0.12, 0.15, meta=500_000)
    b = simulate(0, 1000, 20, 0.12, 0.15, meta=500_000)
    assert a == b
    assert a.valor_final_p10 < a.valor_final_p50 < a.valor_final_p90
    assert len(a.serie_anual) == 20
    assert 0 <= a.prob_atingir_meta <= 1


def test_meta_maior_tem_probabilidade_menor():
    baixa = simulate(0, 1000, 15, 0.12, 0.15, meta=200_000).prob_atingir_meta
    alta = simulate(0, 1000, 15, 0.12, 0.15, meta=800_000).prob_atingir_meta
    assert baixa > alta
