import pytest

from finai.knowledge import taxes
from finai.portfolio.fixed_income import ComparacaoEntrada, OpcaoRendaFixa, comparar


def test_tabela_ir_regressiva():
    assert taxes.ir_rate(100) == 0.225
    assert taxes.ir_rate(300) == 0.20
    assert taxes.ir_rate(700) == 0.175
    assert taxes.ir_rate(1000) == 0.15


def test_iof():
    assert taxes.iof_rate(1) == 0.96
    assert taxes.iof_rate(29) == 0.03
    assert taxes.iof_rate(30) == 0


def test_poupanca_com_selic_alta_rende_meio_por_cento_ao_mes():
    assert taxes.poupanca_annual_rate(0.15) == pytest.approx(1.005**12 - 1)


def test_lci_isenta_vs_cdb_tributado():
    entrada = ComparacaoEntrada(
        valor=10_000,
        prazo_dias=800,
        opcoes=[
            OpcaoRendaFixa(produto="cdb", indexador="pos_cdi", taxa=110),
            OpcaoRendaFixa(produto="lci", indexador="pos_cdi", taxa=90),
            OpcaoRendaFixa(produto="poupanca", indexador="poupanca"),
        ],
    )
    res = comparar(entrada, cdi=0.12, ipca=0.045, selic=0.1225)
    por_produto = {r.produto: r for r in res}
    # CDB 110% com IR de 15% ≈ 93,5% do CDI líquido > LCI 90%
    assert por_produto["cdb"].valor_liquido_final > por_produto["lci"].valor_liquido_final
    assert por_produto["lci"].imposto_renda == 0 and por_produto["lci"].isento_ir
    assert por_produto["poupanca"].valor_liquido_final < por_produto["lci"].valor_liquido_final
    assert res[0].produto == "cdb"  # ordenado pelo valor líquido
    assert por_produto["cdb"].equivalente_percentual_cdi_liquido == pytest.approx(110, rel=0.02)
