from finai.portfolio.planner import montar_carteira
from finai.domain.models import PerfilInvestidor, Preferencias

pref = Preferencias(evitar_cripto=False, evitar_exterior=False, foco_renda_passiva=False, prefere_isentos_ir=False)
for idade in [60, 65, 70, 75]:
    perfil = PerfilInvestidor(idade=idade, renda_mensal=10000, despesas_mensais=5000, tipo_renda='clt', dependentes=0, patrimonio_investido=200000, aporte_mensal=2000, possui_dividas_caras=False, perfil_risco='moderado', experiencia='intermediario', objetivos=[{'nome': 'Aposentadoria', 'valor_alvo': 1000000, 'prazo_anos': 30, 'prioridade': 'alta'}], preferencias=pref, metodo='modelo')
    c = montar_carteira(perfil)
    risky = sum(l.percentual for l in c.alocacao_consolidada if l.classe in ('acoes_br', 'fii', 'internacional', 'cripto', 'multimercado'))
    age_alerts = [a for a in c.alertas if 'idade' in a.lower() or 'renda vari' in a.lower()]
    print('Idade {}: renda_variavel={:.1f}% | alertas={}'.format(idade, risky, age_alerts))