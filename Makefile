.DEFAULT_GOAL := help
.PHONY: help install setup requirements test cov lint format typecheck check chat web api perfil exemplo docker-build docker-up clean

help: ## Lista os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependências (incluindo extras e ferramentas de dev)
	uv sync --all-extras

setup: install ## Primeira configuração: dependências, .env e hooks do git (se houver repositório)
	@test -f .env || (cp .env.example .env && echo "Criado .env - preencha ANTHROPIC_API_KEY")
	@if [ -d .git ]; then uv run pre-commit install; else echo "Dica: depois de 'git init', rode 'uv run pre-commit install'"; fi

requirements: ## Regenera o requirements.txt (para quem usa pip) a partir do uv.lock
	uv export --all-extras --no-hashes --no-emit-project --format requirements-txt -o requirements.txt

test: ## Roda os testes
	uv run pytest

cov: ## Testes com relatório de cobertura
	uv run pytest --cov=finai --cov-report=term-missing

lint: ## Verifica estilo e erros comuns
	uv run ruff check .
	uv run ruff format --check .

format: ## Formata o código
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Checagem de tipos
	uv run mypy src

check: lint typecheck test ## Tudo que o CI roda

chat: ## Conversa com o assistente no terminal
	uv run finai chat

web: ## Interface web (http://localhost:8501)
	uv run streamlit run app/streamlit_app.py

api: ## API REST com recarga automática (http://localhost:8000/docs)
	uv run uvicorn finai.api.main:app --reload

perfil: ## Questionário de perfil de investidor (sem IA)
	uv run finai perfil

exemplo: ## Monta a carteira do perfil de exemplo (sem IA)
	uv run finai carteira examples/perfil_exemplo.json

docker-build: ## Constrói a imagem Docker
	docker compose build

docker-up: ## Sobe API + interface web via Docker
	docker compose up

clean: ## Remove caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
