from finai.agent.engine import FinAIAgent
from finai.config import Settings

agent = FinAIAgent(settings=Settings(refusal_fallback=False))
print('Agent created')

# Test 1: Simple question
print('\n=== Test 1: Simple question ===')
for event in agent.chat('Ola, o que e CDI?'):
    if event.kind == 'text' and event.data:
        print(event.data.get('text', ''), end='')
    elif event.kind == 'tool_start':
        print(f'\n[TOOL: {event.data}]')
    elif event.kind == 'tool_result':
        print(f'[RESULT: OK]')

print('\n\n=== Test 2: Portfolio question ===')
for event in agent.chat('Tenho 30 anos, renda 10000, gasto 5000, CLT, 50k investidos, aporto 1000/mes, perfil moderado, quero montar carteira para aposentadoria em 25 anos'):
    if event.kind == 'text' and event.data:
        print(event.data.get('text', ''), end='')
    elif event.kind == 'tool_start':
        print(f'\n[TOOL: {event.data}]')
    elif event.kind == 'tool_result':
        print(f'[RESULT: OK]')

print('\n\n=== Test 3: Follow-up ===')
for event in agent.chat('E se eu mudar para perfil conservador?'):
    if event.kind == 'text' and event.data:
        print(event.data.get('text', ''), end='')
    elif event.kind == 'tool_start':
        print(f'\n[TOOL: {event.data}]')
    elif event.kind == 'tool_result':
        print(f'[RESULT: OK]')