"""Configuração dos testes: força modo offline (sem rede) e limpa caches."""

import os

os.environ["FINAI_OFFLINE"] = "true"
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")  # o cliente real nunca é chamado nos testes

import pytest

from finai.config import get_settings


@pytest.fixture(autouse=True)
def _offline_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
