from finai.domain.suitability import avaliar_perfil
from finai.domain.models import RespostasSuitability

testes = [
    ('Conservador puro', RespostasSuitability(
        objetivo='preservar_capital', horizonte='ate_1_ano',
        reacao_queda_20='vender_tudo', conhecimento='nenhum',
        experiencia_anos='nunca', estabilidade_renda='instavel',
        necessidade_liquidez='alta'
    )),
    ('Moderado tipico', RespostasSuitability(
        objetivo='crescimento_moderado', horizonte='5_a_10_anos',
        reacao_queda_20='manter', conhecimento='intermediario',
        experiencia_anos='2_a_5', estabilidade_renda='estavel',
        necessidade_liquidez='baixa'
    )),
    ('Arrojado ideal', RespostasSuitability(
        objetivo='maximizar_crescimento', horizonte='mais_de_10_anos',
        reacao_queda_20='comprar_mais', conhecimento='avancado',
        experiencia_anos='mais_de_5', estabilidade_renda='muito_estavel',
        necessidade_liquidez='nenhuma'
    )),
    ('Travas: conhecimento nenhum', RespostasSuitability(
        objetivo='maximizar_crescimento', horizonte='mais_de_10_anos',
        reacao_queda_20='comprar_mais', conhecimento='nenhum',
        experiencia_anos='nunca', estabilidade_renda='muito_estavel',
        necessidade_liquidez='nenhuma'
    )),
    ('Travas: liquidez alta', RespostasSuitability(
        objetivo='maximizar_crescimento', horizonte='mais_de_10_anos',
        reacao_queda_20='comprar_mais', conhecimento='avancado',
        experiencia_anos='mais_de_5', estabilidade_renda='muito_estavel',
        necessidade_liquidez='alta'
    )),
    ('Limiar conservador/moderado', RespostasSuitability(
        objetivo='preservar_capital', horizonte='1_a_3_anos',
        reacao_queda_20='manter', conhecimento='basico',
        experiencia_anos='menos_de_2', estabilidade_renda='estavel',
        necessidade_liquidez='media'
    )),
    ('Limiar moderado/arrojado', RespostasSuitability(
        objetivo='crescimento_moderado', horizonte='5_a_10_anos',
        reacao_queda_20='comprar_mais', conhecimento='intermediario',
        experiencia_anos='2_a_5', estabilidade_renda='muito_estavel',
        necessidade_liquidez='baixa'
    )),
]

for nome, respostas in testes:
    resultado = avaliar_perfil(respostas)
    travas = str(resultado.travas_aplicadas).replace('→', '->') if resultado.travas_aplicadas else "none"
    exp = str(resultado.experiencia_sugerida).replace('→', '->')
    print(f'{nome}: {resultado.perfil.upper()} (pontuacao {resultado.pontuacao:.1f}) | travas: {travas} | exp_sugerida: {exp}')