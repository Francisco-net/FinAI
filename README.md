# FinAI: assistente de investimentos com IA

> Um agente de inteligência artificial que entende a sua situação financeira e os seus objetivos
> e monta uma **carteira de investimentos personalizada para o mercado brasileiro**, explicada passo a passo.

> ⚠️ **Ferramenta educacional.** O FinAI não constitui recomendação de investimento. Decisões
> relevantes devem ser validadas com um profissional certificado (CFP, CEA ou consultor registrado na CVM).

## Sumário

1. [Sobre o projeto](#sobre-o-projeto)
2. [Funcionalidades](#funcionalidades)
3. [Exemplo de resultado](#exemplo-de-resultado)
4. [Como funciona](#como-funciona)
5. [Tecnologias](#tecnologias)
6. [Como rodar](#como-rodar)
7. [Desenvolvimento](#desenvolvimento)
8. [Estrutura de pastas](#estrutura-de-pastas)
9. [API REST](#api-rest)
10. [Configuração](#configuração)
11. [Como estender](#como-estender)
12. [Solução de problemas](#solução-de-problemas)
13. [Próximos passos](#próximos-passos)
14. [Avisos regulatórios](#avisos-regulatórios)

---

## Sobre o projeto

Começar a investir costuma esbarrar em três problemas: excesso de jargão, excesso de opções e falta
de um plano que ligue o dinheiro aos objetivos da pessoa. Quem já investe, por outro lado, gasta tempo
montando planilhas, comparando produtos e recalculando aportes.

O FinAI resolve os dois lados com uma conversa:

- **Para iniciantes:** linguagem simples, explicação de cada termo e um plano concreto do que fazer todo mês.
- **Para investidores experientes:** respostas diretas e técnicas, com métricas, premissas e
  comparações prontas, economizando tempo e esforço.

A pessoa conta sua situação (idade, renda, gastos, quanto já tem, quanto consegue investir e o que
quer conquistar) e o assistente devolve uma **carteira conectada ao seu cenário**: reserva de
emergência, uma "caixinha" para cada objetivo, quanto aportar em cada uma, em quais tipos de
investimento e por quê.

## Funcionalidades

| Funcionalidade | O que faz |
|---|---|
| **Perfil de investidor (suitability)** | Questionário que classifica em conservador, moderado ou arrojado, com "travas" de segurança (ex.: quem venderia tudo numa queda de 20% não pode ser arrojado) |
| **Carteira por objetivos** | Reserva de emergência primeiro; cada objetivo vira uma caixinha com investimentos adequados ao prazo; cálculo do aporte mensal necessário para cada meta |
| **Personalização** | Ajustes por experiência (iniciante sem cripto e sem crédito privado), idade (menos risco perto da aposentadoria) e preferências (sem cripto, sem exterior, foco em renda passiva, isentos de IR) |
| **Otimização de carteira** | Método "otimizado" (Markowitz) com volatilidade-alvo por perfil, como alternativa às carteiras-modelo |
| **Simulação de patrimônio** | Projeção Monte Carlo com cenários pessimista, mediano e otimista, em reais de hoje, e probabilidade de atingir a meta |
| **Comparador de renda fixa** | CDB × LCI/LCA × Tesouro × debêntures × poupança, com rentabilidade **líquida** de IR, IOF e custos |
| **Dados de mercado** | Selic, CDI, IPCA e dólar do Banco Central; cotações de ações, ETFs e FIIs |
| **Glossário** | Explicação simples de mais de 35 conceitos (CDI, FGC, come-cotas, marcação a mercado...) |
| **Alertas** | Dívidas caras, gastos maiores que a renda, aporte insuficiente, limite do FGC |

Tudo isso está disponível em três interfaces: **chat no terminal**, **interface web** e **API REST**.

## Exemplo de resultado

Resultado real do motor de carteiras para o perfil de [`examples/perfil_exemplo.json`](examples/perfil_exemplo.json):
29 anos, CLT, renda de R$ 7.500, gastos de R$ 4.500, R$ 35.000 investidos, aporte de R$ 1.800/mês,
perfil moderado, iniciante, sem cripto e preferindo produtos isentos de IR.

```bash
uv run finai carteira examples/perfil_exemplo.json
```

**Caixinhas por objetivo**

| Caixinha | Prazo | Meta | Alocado hoje | Aporte/mês | Alocação | Status |
|---|---|---|---|---|---|---|
| Reserva de emergência | - | R$ 27.000 | R$ 27.000 | R$ 0 | Pós-fixado 100% | Completa |
| Entrada do apartamento | 5 anos | R$ 90.000 | R$ 8.000 | R$ 1.117,26 | Pós 45% · IPCA+ 30% · Pré 10% · Multi 5% · FII 4% · Ações BR 3% · Exterior 3% | No caminho: o aporte cobre a meta |
| Viagem para a Europa | 2 anos | R$ 18.000 | R$ 0 | R$ 682,74 | Pós 70% · Pré 30% | Aporte insuficiente: faltam R$ 23,55/mês |
| Aposentadoria | 30 anos | R$ 2.000.000 | R$ 0 | R$ 0 | IPCA+ 28% · Pós 25% · Ações BR 12% · Exterior 12% · Multi 8% · FII 8% · Pré 7% | Aporte insuficiente: faltam R$ 1.605,61/mês |

Além da tabela, o resultado traz alertas ("para a aposentadoria seriam necessários R$ 1.605,61/mês;
opções: aumentar o aporte, estender o prazo ou ajustar a meta"), a alocação consolidada com exemplos
de produtos (Tesouro Selic, LCI/LCA, Tesouro IPCA+, BOVA11, IVVB11...), métricas de risco e os próximos
passos práticos. No chat, o assistente usa esse mesmo resultado para explicar o plano e discutir
ajustes, como "e se eu adiar a viagem em 6 meses?".

## Como funciona

```
Usuário ──► Agente Claude (conversa, entende necessidades, explica)
               │  chama ferramentas (tool use)
               ▼
   ┌──────────────────────── Ferramentas determinísticas ────────────────────────┐
   │ avaliar_perfil_investidor   suitability (conservador/moderado/arrojado)     │
   │ montar_carteira             planejamento por objetivos + alocação           │
   │ simular_patrimonio          Monte Carlo em R$ de hoje                       │
   │ comparar_renda_fixa         CDB × LCI × Tesouro... líquido de IR/IOF        │
   │ consultar_indicadores       Selic, CDI, IPCA, dólar (Banco Central)         │
   │ consultar_cotacoes          ações/ETFs/FIIs (yfinance)                      │
   │ explicar_conceito           glossário de termos do mercado                  │
   │ calcular_reserva_emergencia / listar_classes_de_ativos                      │
   └─────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
   Base de conhecimento versionada (src/finai/data/*.yaml):
   premissas de retorno/risco, correlações, carteiras-modelo, tributação, FGC, glossário
```

**Princípio central:** o modelo de linguagem conversa, entende e explica; os **números** (alocação,
aportes, projeções, impostos) vêm sempre de código Python testado. Isso evita que a IA invente
percentuais e torna cada resultado auditável e reproduzível.

### O planejador (`montar_carteira`)

1. **Saúde financeira:** alerta sobre dívidas caras e gastos maiores que a renda.
2. **Reserva de emergência primeiro:** de 3 a 12 meses de despesas conforme o tipo de renda
   (+1 mês por dependente), 100% em pós-fixado com liquidez diária.
3. **Uma caixinha por objetivo**, com alocação conforme o **prazo**:
   - até 1 ano: 100% pós-fixado; até 2 anos: pós-fixado + prefixado;
   - de 2 a 5 anos: um degrau mais conservador que o perfil;
   - acima de 5 anos: carteira do perfil (modelo ou otimizada).
4. **Distribuição do patrimônio por prioridade** e cálculo do **aporte mensal necessário** para
   cada objetivo, com alertas quando o aporte não fecha a conta.
5. **Ajustes ao investidor:** experiência, idade, preferências e tamanho do patrimônio.
6. **Consolidação:** alocação final, exemplos de produtos, métricas (retorno real esperado,
   volatilidade, perda em um ano ruim), alerta de FGC, plano de aportes e próximos passos.

### O agente

O agente usa o **Claude Opus 5.5** com streaming (a resposta aparece enquanto é escrita),
raciocínio adaptativo e cache de prompt (o prompt de sistema e as ferramentas são reaproveitados
entre mensagens, o que reduz custo e latência). O loop do agente fica em
[`src/finai/agent/engine.py`](src/finai/agent/engine.py) e é independente de interface: o terminal,
a API e a interface web consomem os mesmos eventos.

## Tecnologias

| Camada | Ferramenta | Por quê |
|---|---|---|
| Linguagem | **Python 3.11+** | Ecossistema líder em dados, finanças quantitativas e IA |
| LLM / agente | **Claude Opus 5.5** via SDK oficial `anthropic` | Uso de ferramentas, streaming, raciocínio adaptativo, cache de prompt |
| Domínio / validação | Pydantic v2 | Modelos tipados que também geram o JSON Schema das ferramentas |
| Cálculo | NumPy, SciPy, pandas | Otimização de carteira, Monte Carlo, métricas de risco |
| Dados de mercado | API SGS do Banco Central, yfinance | Indicadores oficiais e cotações da B3 |
| Interfaces | Typer + Rich (terminal), FastAPI (API/SSE), Streamlit + Plotly (web) | Um motor, três formas de uso |
| Qualidade | uv, ruff, mypy, pytest, pre-commit, GitHub Actions | Ambiente reproduzível e integração contínua |
| Execução | Docker / docker compose, Dev Container | O mesmo ambiente na sua máquina, em contêiner e no VS Code/Codespaces |

## Como rodar

### Pré-requisitos

- Conexão com a internet na primeira instalação (para baixar o Python e as bibliotecas).
- Uma chave da API da Anthropic, criada em https://platform.claude.com, na seção "API Keys".
  Ela só é necessária para o chat com IA; o uso é cobrado conforme o consumo.

Você **não** precisa instalar o Python antes: o script de instalação cuida disso.

### 1. Obter o código

Descompacte o `FinAI.zip` ou, se o projeto já estiver no seu GitHub, clone o repositório:
```bash
git clone https://github.com/Francisco-net/FinAI.git
```

### 2. Instalar tudo com um comando

Abra um terminal **dentro da pasta do projeto** e rode o script do seu sistema.

Windows (PowerShell):
```bash
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Mac ou Linux:
```bash
bash scripts/setup.sh
```

O script instala o [uv](https://docs.astral.sh/uv/) (se ainda não tiver), o Python 3.13 e **todas as
dependências nas versões exatas travadas no `uv.lock`**, cria o arquivo `.env` e roda os testes
para confirmar que está tudo funcionando. As bibliotecas ficam na pasta `.venv` do projeto, sem
interferir no resto do seu computador. Pode rodar o script de novo sempre que quiser.

### 3. Configurar a chave da API

Abra o arquivo `.env` em um editor de texto e preencha a linha `ANTHROPIC_API_KEY=` com a sua chave.
O `.env` está no `.gitignore` e nunca deve ser enviado ao GitHub.

### 4. Rodar

**Interface web** (recomendado): abre em http://localhost:8501
```bash
uv run streamlit run app/streamlit_app.py
```

**Chat no terminal** (digite `nova` para recomeçar e `sair` para encerrar):
```bash
uv run finai chat
```

**API REST**: documentação interativa em http://localhost:8000/docs
```bash
uv run uvicorn finai.api.main:app --reload
```

Se o terminal disser que o comando `uv` não existe, feche e abra o terminal (o instalador acabou de
adicioná-lo ao PATH).

### Instalação manual (sem o script)

1. Instale o uv. Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`. Windows (PowerShell):
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
2. Na pasta do projeto, instale as dependências:
   ```bash
   uv sync --all-extras
   ```
3. Copie `.env.example` para `.env` e preencha a chave.

### Alternativa com pip (sem uv)

Requer Python 3.11 ou mais recente já instalado. O `requirements.txt` tem as mesmas versões do `uv.lock`.

```bash
python -m venv .venv
```
Ative o ambiente. Windows: `.venv\Scripts\activate` · Mac/Linux: `source .venv/bin/activate`
```bash
pip install -r requirements.txt
```
```bash
pip install -e . --no-deps
```
Com o ambiente ativado, rode os comandos sem o prefixo `uv run` (ex.: `streamlit run app/streamlit_app.py`, `finai chat`).

### Versionar no seu GitHub

O projeto vem sem histórico de git, para os commits serem seus. Na pasta do projeto:
```bash
git init -b main
```
```bash
uv run pre-commit install
```
```bash
git add .
```
```bash
git commit -m "Primeiro commit do FinAI"
```
```bash
git remote add origin https://github.com/Francisco-net/FinAI.git
```
```bash
git push -u origin main
```
O `pre-commit` roda lint, formatação e checagem de tipos automaticamente antes de cada commit.

### Sem chave de API

O motor de carteiras funciona sem IA e sem internet:

```bash
uv run finai carteira examples/perfil_exemplo.json
```
```bash
uv run finai perfil
```
```bash
uv run finai simular --inicial 10000 --aporte 1500 --anos 25 --perfil moderado --meta 1500000
```
```bash
uv run finai indicadores
```

Na interface web, a aba **Planejador sem IA** também funciona sem chave.

### Atalhos com `make` (Mac/Linux)

Com o `make` instalado, `make setup` instala as dependências e cria o `.env`; depois:
`make web`, `make chat`, `make api` ou `make exemplo`. Rode `make help` para ver todos os comandos.

### Com Docker

Depois de criar o `.env`:
```bash
docker compose up
```
A interface web abre em http://localhost:8501 e a API em http://localhost:8000.

### Com Dev Container (VS Code / GitHub Codespaces)

Abra a pasta no VS Code e escolha **Reopen in Container**, ou crie um Codespace pelo GitHub.
O ambiente instala tudo automaticamente e já encaminha as portas 8000 e 8501.

## Desenvolvimento

| Comando | O que faz |
|---|---|
| `uv run pytest` | Roda os testes |
| `uv run pytest --cov=finai --cov-report=term-missing` | Testes com cobertura |
| `uv run ruff check .` | Verifica estilo e erros comuns |
| `uv run ruff format .` | Formata o código |
| `uv run mypy src` | Checagem de tipos |
| `make check` | Lint + tipos + testes (o mesmo que o CI roda) |

Os testes não acessam a internet nem a API da Anthropic: o loop do agente é testado com um cliente
falso ([`tests/test_engine.py`](tests/test_engine.py)), então rodar os testes não tem custo.
O CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) roda lint, tipos e testes a cada push
e pull request.

## Estrutura de pastas

```
src/finai/
  agent/        engine.py (loop de streaming + ferramentas), tools.py, prompts.py
  domain/       models.py (modelos Pydantic), suitability.py (perfil de investidor)
  portfolio/    planner.py (planejador por objetivos), allocation.py, optimizer.py,
                metrics.py, simulation.py, reserve.py, fixed_income.py
  knowledge/    assets.py (premissas), taxes.py (IR/IOF/poupança), glossary.py
  market/       bcb.py (Banco Central), quotes.py (cotações), cache.py
  data/         assumptions.yaml, glossario.yaml   ← premissas de mercado versionadas
  api/          main.py (FastAPI)
  cli.py        comandos de terminal
  config.py     configuração via .env (prefixo FINAI_)
app/streamlit_app.py   interface web
scripts/               setup.sh (Mac/Linux) e setup.ps1 (Windows): instalação em um comando
examples/              perfil de exemplo
tests/                 testes automatizados
uv.lock                versões exatas de todas as dependências
requirements.txt       as mesmas versões, para quem usa pip
```

## API REST

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status da API |
| GET | `/indicadores` | Selic, CDI, IPCA e dólar |
| POST | `/perfil` | Perfil de investidor a partir das respostas do questionário |
| POST | `/carteira` | Carteira a partir de um `PerfilInvestidor` |
| POST | `/simulacao` | Projeção Monte Carlo |
| POST | `/renda-fixa/comparar` | Comparação líquida de produtos de renda fixa |
| POST | `/chat/{sessao}` | Conversa com o agente (resposta em Server-Sent Events) |
| GET | `/chat/{sessao}/carteira` | Última carteira montada na sessão |
| DELETE | `/chat/{sessao}` | Encerra a sessão |

Exemplo de conversa pela API:

```bash
curl -N -X POST localhost:8000/chat/minha-sessao -H 'Content-Type: application/json' -d '{"mensagem": "Tenho 30 anos, ganho 8 mil, gasto 5 mil e quero me aposentar cedo"}'
```

## Configuração

Todas as opções ficam no `.env` (veja [`.env.example`](.env.example)):

| Variável | Padrão | Descrição |
|---|---|---|
| `ANTHROPIC_API_KEY` | (nenhum) | Chave da API da Anthropic |
| `FINAI_MODEL` | `claude-opus-5-5` | Modelo do agente |
| `FINAI_EFFORT` | `medium` | Profundidade de raciocínio: `low`, `medium`, `high`, `xhigh` ou `max` (mais alto = respostas mais cuidadosas, porém mais caras e lentas) |
| `FINAI_THINKING_DISPLAY` | `summarized` | Mostra um resumo do raciocínio do modelo na interface |
| `FINAI_REFUSAL_FALLBACK` | `true` | Se o modelo recusar um pedido pelos filtros de segurança, a Anthropic refaz o pedido em outro modelo automaticamente |
| `FINAI_OFFLINE` | `false` | Não acessa a internet (usa valores de referência para os indicadores) |

## Como estender

- **Atualizar premissas de mercado** (retornos esperados, correlações, carteiras-modelo, impostos):
  edite [`src/finai/data/assumptions.yaml`](src/finai/data/assumptions.yaml) e atualize
  `versao`/`revisado_em`. Os testes validam a consistência.
- **Nova ferramenta para o agente:** crie o modelo de entrada (Pydantic) e a função, e registre
  em `TOOLS` ([`src/finai/agent/tools.py`](src/finai/agent/tools.py)). O schema é gerado automaticamente.
- **Novo conceito no glossário:** adicione em [`src/finai/data/glossario.yaml`](src/finai/data/glossario.yaml).
- **Comportamento do agente:** [`src/finai/agent/prompts.py`](src/finai/agent/prompts.py). Mantenha
  o texto estável (sem datas ou valores dinâmicos), pois ele é cacheado.

## Solução de problemas

| Problema | Solução |
|---|---|
| "Nenhuma credencial da Anthropic encontrada" | Crie o `.env` (passo 4), preencha `ANTHROPIC_API_KEY` e rode o comando a partir da pasta do projeto |
| "Chave da API inválida" | Confira se a chave foi copiada inteira, sem espaços, e se está ativa em https://platform.claude.com |
| Indicadores com aviso "valor de referência offline" | O Banco Central não respondeu (rede, firewall ou instabilidade). O FinAI segue funcionando com valores de referência; tente de novo mais tarde |
| `uv: command not found` | Feche e abra o terminal depois de instalar o uv |
| Windows bloqueia o script ("execução de scripts foi desabilitada") | Rode exatamente `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1`, que libera só esse script |
| `make: command not found` (Windows) | Use os comandos `uv run ...` da seção [Como rodar](#como-rodar) |
| Porta 8501 ou 8000 em uso | Use outra porta: `--server.port 8502` no Streamlit ou `--port 8001` no uvicorn |
| Cotações indisponíveis | O yfinance depende do Yahoo Finance; pode haver limite de requisições. Tente novamente em alguns minutos |

## Próximos passos

- Persistência de usuários, perfis e conversas (PostgreSQL) com autenticação
- Importar a carteira atual (extrato da B3 ou da corretora) e sugerir **rebalanceamento**
- Expectativas do boletim Focus (API Olinda do Banco Central) para premissas dinâmicas
- Previdência privada (PGBL × VGBL) e planejamento tributário
- Avaliações automáticas (evals) da qualidade das respostas do agente
- Monitoramento e alertas (desvio de alocação, vencimentos, metas atrasadas)

## Avisos regulatórios

O FinAI aplica conceitos de suitability (Resolução CVM 30/2021) e fornece **informação educacional**.
A recomendação personalizada de valores mobiliários no Brasil é atividade regulada (Resoluções
CVM 19 e 20). Antes de oferecer o produto a terceiros, avalie com assessoria jurídica o
enquadramento regulatório, a LGPD (dados financeiros pessoais) e os avisos necessários.

As premissas de mercado são estimativas de cenário de longo prazo e não garantem retornos futuros.
As regras tributárias mudam com frequência: confirme a legislação vigente antes de decisões relevantes.
