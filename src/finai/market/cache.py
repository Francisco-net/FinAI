"""Cache em memória com expiração (evita bater nas APIs de mercado a cada pergunta)."""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def ttl_cache(seconds_getter: Callable[[], float]) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        store: dict[tuple, tuple[float, Any]] = {}
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with lock:
                hit = store.get(key)
                if hit and now - hit[0] < seconds_getter():
                    return hit[1]
            value = func(*args, **kwargs)
            with lock:
                store[key] = (now, value)
            return value

        wrapper.cache_clear = store.clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
