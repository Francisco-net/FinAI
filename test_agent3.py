from finai.agent.engine import FinAIAgent
from finai.config import Settings

agent = FinAIAgent(settings=Settings(refusal_fallback=False))
print('Agent created successfully')

# First message
response1 = 'Tenho 30 anos, renda 8000, despesas 5000, sou CLT, tenho 35000 investidos, aporte 1800 por mes, sem dividas, perfil moderado, iniciante, quero apartamento 90000 em 5 anos prioridade alta, viagem 18000 em 2 anos prioridade baixa, aposentadoria 2000000 em 30 anos prioridade media, nao evito cripto, nao evito exterior, sem foco renda passiva, prefiro isentos IR'

print("=== PRIMEIRA MENSAGEM ===")
for ev in agent.chat(response1):
    if ev.kind == 'text':
        print(f'  {ev.text}', end='')
    elif ev.kind == 'tool_start':
        print(f'\n[TOOL START: {ev.tool_name}]')
    elif ev.kind == 'tool_result':
        print(f'\n[TOOL RESULT: {ev.tool_name} - {"ERROR" if ev.is_error else "OK"}]')
    elif ev.kind == 'done':
        print('\n[DONE]')
    elif ev.kind == 'thinking':
        pass

print("\n\n=== SEGUNDA MENSAGEM (resposta suitability) ===")
# Second message - answer suitability
response2 = 'crescimento_moderado, 3_a_5_anos, manter, basico, menos_de_2, estavel, baixa'

for ev in agent.chat(response2):
    if ev.kind == 'text':
        print(f'  {ev.text}', end='')
    elif ev.kind == 'tool_start':
        print(f'\n[TOOL START: {ev.tool_name}]')
    elif ev.kind == 'tool_result':
        print(f'\n[TOOL RESULT: {ev.tool_name} - {"ERROR" if ev.is_error else "OK"}]')
    elif ev.kind == 'done':
        print('\n[DONE]')
    elif ev.kind == 'thinking':
        pass