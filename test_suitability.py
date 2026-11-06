from finai.domain.suitability import avaliar_perfil
from finai.domain.models import RespostasSuitability

testes = [
    ('Conservador tipico', RespostasSuitability(
        objetivo='preservar_capital',
        horizonte='ate_1_ano',
        reacao_queda_20='vender_tudo',
        conhecimento='nenhum',
        experiencia_anos='nunca',
        estabilidade_renda='instavel',
        necessidade_liquidez='alta'
    )),
    ('Moderado equilibrado', RespostasSuitability(
        objetivo='crescimento_moderado',
        horizonte='3_a_5_anos',
        reacao_queda_20='manter',
        conhecimento='basico',
        experiencia_anos='menos_de_2',
        estabilidade_renda='estavel',
        necessidade_liquidez='media'
    )),
    ('Arrojado experiente', RespostasSuitability(
        objetivo='maximizar_crescimento',
        horizonte='mais_de_10_anos',
        reacao_queda_20='comprar_mais',
        conhecimento='avancado',
        experiencia_anos='mais_de_5',
        estabilidade_renda='muito_estavel',
        necessidade_liquidez='nenhuma'
    )),
    ('Renda complementar com horizonte medio', RespostasSuitability(
        objetivo='renda_complementar',
        horizonte='5_a_10_anos',
        reacao_queda_20='vender_parte',
        conhecimento='intermediario',
        experiencia_anos='2_a_5',
        estabilidade_renda='variavel',
        necessidade_liquidez='baixa'
    )),
]

for nome, respostas in testes:
    resultado = avaliar_perfil(respostas)
    print(f'{nome}: {resultado.perfil.upper()} (pontuação {resultado.pontuacao})')
    if resultado.travas_aplicadas:
        print(f'  Travas: {"; ".join(resultado.travas_aplicadas)}')
    print()