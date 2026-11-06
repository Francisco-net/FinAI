from finai.agent.engine import FinAIAgent
from finai.config import Settings

agent = FinAIAgent(settings=Settings(refusal_fallback=False))
print('Agent created')

# First message
for ev in agent.chat('Tenho 30 anos, renda 8000, despesas 5000, CLT, 35000 investidos, aporte 1800, sem dividas, moderado, iniciante, apartamento 90000 em 5 anos alta, viagem 18000 em 2 anos baixa, aposentadoria 2000000 em 30 anos media, nao evito cripto nem exterior, prefiro isentos'):
    if ev.kind in ('text', 'tool_start', 'tool_result', 'done'):
        txt = ev.text[:30] if ev.text else ''
        print(f'{ev.kind}: {ev.tool_name or txt}')