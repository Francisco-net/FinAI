#!/usr/bin/env bash
# Instala tudo que o FinAI precisa (Mac/Linux): uv, Python 3.13 e todas as dependências
# nas versões exatas travadas no uv.lock. Pode ser executado mais de uma vez com segurança.
#
# Uso (dentro da pasta do projeto):  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Instalando o uv (gerenciador de projetos Python)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "==> Instalando o Python e as dependências (versões exatas do uv.lock)..."
uv sync --all-extras --frozen

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Arquivo .env criado. Preencha ANTHROPIC_API_KEY para usar o chat com IA."
fi

if [ -d .git ]; then
  uv run pre-commit install
fi

echo "==> Verificando a instalação (testes automatizados)..."
uv run pytest

cat <<'MSG'

Instalação concluída!

  Interface web:      uv run streamlit run app/streamlit_app.py   (http://localhost:8501)
  Chat no terminal:   uv run finai chat
  API REST:           uv run uvicorn finai.api.main:app --reload  (http://localhost:8000/docs)
  Sem chave de API:   uv run finai carteira examples/perfil_exemplo.json

Para o chat com IA, preencha ANTHROPIC_API_KEY no arquivo .env.
MSG
