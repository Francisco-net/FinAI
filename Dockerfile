
,# Imagem única para a API (padrão) e para a interface web (via docker compose).
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependências primeiro (camada cacheada enquanto o lockfile não muda)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --all-extras --no-install-project

COPY src ./src
COPY app ./app
RUN uv sync --frozen --no-dev --all-extras

RUN useradd --create-home finai && chown -R finai /app
USER finai

EXPOSE 8000 8501
CMD ["uvicorn", "finai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
