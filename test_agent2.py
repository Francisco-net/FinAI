from finai.agent.engine import FinAIAgent
from finai.config import Settings

agent = FinAIAgent(settings=Settings(refusal_fallback=False))
print('Agent created successfully')
print('Tool definitions count:', len(agent._tools))

# Complete profile
response = 'Tenho 30 anos, renda 8000, despesas 5000, sou CLT, tenho 35000 investidos, aporte 1800 por mes, sem dividas, perfil moderado, iniciante, quero apartamento 90000 em 5 anos prioridade alta, viagem 18000 em 2 anos prioridade baixa, aposentadoria 2000000 em 30 anos prioridade media, nao evito cripto, nao evito exterior, sem foco renda passiva, prefiro isentos IR'

for ev in agent.chat(response):
    if ev.kind == 'text':
        print(f'Event: {ev.kind} - {ev.text[:50]}')
    elif ev.kind == 'tool_start':
        print(f'Event: {ev.kind} - {ev.tool_name}')
    elif ev.kind == 'tool_result':
        print(f'Event: {ev.kind} - {ev.tool_name} - {"ERROR" if ev.is_error else "OK"}')
    elif ev.kind == 'done':
        print(f'Event: {ev.kind}')
    else:
        print(f'Event: {ev.kind}')