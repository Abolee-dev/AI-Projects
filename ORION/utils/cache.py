import hashlib
import json
from typing import Any

import diskcache

from utils.config import settings

_cache: diskcache.Cache | None = None


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        _cache = diskcache.Cache(str(settings.cache_dir))
    return _cache


def cache_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def get(key: str) -> Any | None:
    return _get_cache().get(key)


def set(key: str, value: Any, ttl: int = 3600) -> None:
    _get_cache().set(key, value, expire=ttl)
