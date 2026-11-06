from finai.domain.suitability import avaliar_perfil
from finai.domain.models import RespostasSuitability

testes = [
    ('Limiar conservador/moderado', RespostasSuitability(
        objetivo='preservar_capital',
        horizonte='1_a_3_anos',
        reacao_queda_20='manter',
        conhecimento='basico',
        experiencia_anos='menos_de_2',
        estabilidade_renda='estavel',
        necessidade_liquidez='media'
    )),
    ('Limiar moderado/arrojado', RespostasSuitability(
        objetivo='crescimento_moderado',
        horizonte='5_a_10_anos',
        reacao_queda_20='comprar_mais',
        conhecimento='intermediario',
        experiencia_anos='2_a_5',
        estabilidade_renda='muito_estavel',
        necessidade_liquidez='baixa'
    )),
    ('Travas por conhecimento', RespostasSuitability(
        objetivo='maximizar_crescimento',
        horizonte='mais_de_10_anos',
        reacao_queda_20='comprar_mais',
        conhecimento='nenhum',
        experiencia_anos='nunca',
        estabilidade_renda='muito_estavel',
        necessidade_liquidez='nenhuma'
    )),
    ('Travas por liquidez', RespostasSuitability(
        objetivo='maximizar_crescimento',
        horizonte='mais_de_10_anos',
        reacao_queda_20='comprar_mais',
        conhecimento='avancado',
        experiencia_anos='mais_de_5',
        estabilidade_renda='muito_estavel',
        necessidade_liquidez='alta'
    )),
]

for nome, respostas in testes:
    resultado = avaliar_perfil(respostas)
    print(f'{nome}: {resultado.perfil.upper()} (pontuação {resultado.pontuacao})')
    if resultado.travas_aplicadas:
        print(f'  Travas: {"; ".join(resultado.travas_aplicadas)}')
    print(f'  Experiência sugerida: {resultado.experiencia_sugerida}')
    print()