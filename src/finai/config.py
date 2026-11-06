"""Configuração central, lida de variáveis de ambiente (prefixo FINAI_) e do arquivo .env.

A chave da Anthropic (ANTHROPIC_API_KEY) não é um campo daqui: o SDK a lê do ambiente, para onde o .env é exportado em get_settings().
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINAI_", env_file=".env", extra="ignore")

    # --- LLM ---
    model: str = "claude-opus-5-5"
    # Esforço de raciocínio: low | medium | high | xhigh | max (no Opus 5.5 o padrão da API é medium)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = 32000
    # Mostra um resumo do raciocínio do modelo na interface (útil para transparência/debug)
    thinking_display: Literal["summarized", "omitted"] = "summarized"
    # Se o modelo recusar por política de segurança, refaz a requisição em outro modelo (server-side)
    refusal_fallback: bool = True
    max_tool_iterations: int = 12

    # --- Dados de mercado ---
    bcb_timeout_seconds: float = 8.0
    market_cache_ttl_seconds: int = 3600
    offline: bool = False  # força uso dos valores de fallback (útil em testes/ambientes sem rede)

    # --- Banco de dados ---
    database_url: str = "sqlite+aiosqlite:///./finai.db"
    database_echo: bool = False
    database_pool_disabled: bool = False  # True para serverless (NullPool)


@lru_cache
def get_settings() -> Settings:
    # Exporta o .env para o ambiente: o SDK da Anthropic lê ANTHROPIC_API_KEY de os.environ.
    # Variáveis já definidas no ambiente têm prioridade sobre o arquivo.
    load_dotenv(find_dotenv(usecwd=True), override=False)
    return Settings()
