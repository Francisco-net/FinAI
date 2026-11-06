"""Busca no glossário de conceitos financeiros (tolerante a acentos e variações)."""

from __future__ import annotations

import difflib
import unicodedata

from finai.knowledge.assets import load_glossary


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().strip())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("_", " ").replace("-", " ")


def lookup(term: str) -> dict:
    """Retorna a definição do termo, ou sugestões de termos próximos."""
    glossary = load_glossary()
    query = _normalize(term)

    index: dict[str, str] = {}
    for canonical, entry in glossary.items():
        index[_normalize(canonical)] = canonical
        for syn in entry.get("sinonimos", []):
            index[_normalize(str(syn))] = canonical

    key: str | None = index.get(query)
    if key is None:
        # termo contido em alguma chave/sinônimo (ex.: "tesouro ipca" em "tesouro ipca+")
        key = next((k for alias, k in index.items() if query and query in alias), None)
    if key is None:
        close = difflib.get_close_matches(query, list(index), n=1, cutoff=0.75)
        key = index[close[0]] if close else None

    if key is None:
        suggestions = difflib.get_close_matches(query, list(index), n=5, cutoff=0.4)
        return {
            "encontrado": False,
            "termo": term,
            "sugestoes": sorted({index[s] for s in suggestions}),
            "termos_disponiveis": sorted(glossary),
        }
    return {"encontrado": True, "termo": key, "definicao": glossary[key]["definicao"].strip()}
