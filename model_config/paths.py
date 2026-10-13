"""
Infer cache paths from the model name and cache_base.
Path convention: <cache_base>/<model_name>/within_ske_cache
model_name may contain / (e.g. openai-community/gpt2); just join directly.
"""

import hashlib
import os
import re


def safe_slug(text: str, max_len: int = 40) -> str:
    """Filesystem-safe slug for a target token/word used inside a result filename.

    Target strings can contain '/' (real datasets include tokens like "//", "/(",
    " //" — common in code/programming targets). Naively replacing only spaces
    lets '/' through, which os.path.join then treats as a subdirectory separator
    — or, if the slug starts with '/', as an absolute path that silently discards
    the intended output directory. Every character outside [0-9A-Za-z._-] is
    replaced with '_', the result is length-capped, and a short hash of the
    original (pre-slug) string is appended so distinct targets that collapse to
    the same slug (e.g. "//" and "/(" both becoming "__") stay distinguishable.
    """
    stripped = text.strip()
    digest = hashlib.sha1(stripped.encode("utf-8")).hexdigest()[:8]
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", stripped.replace(" ", "_"))[:max_len]
    return f"{safe or 'empty'}_{digest}"


def within_ske_path(model_name: str, cache_base: str) -> str | None:
    """Return the path to the within_ske_cache directory, or None if it does not exist."""
    p = os.path.join(cache_base, model_name, "within_ske_cache")
    return p if os.path.isdir(p) else None
