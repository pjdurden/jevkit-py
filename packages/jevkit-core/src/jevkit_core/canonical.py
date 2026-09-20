"""RFC 8785 JSON Canonicalization, and the digests jevkit builds on it."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

__all__ = ["canonical_json", "digest", "request_id", "record_id"]


def _check_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"cannot canonicalize non-finite number: {value!r}")
    if isinstance(value, dict):
        for v in value.values():
            _check_finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _check_finite(v)


def canonical_json(value: Any) -> bytes:
    """Serialize to RFC 8785 canonical form.

    Object keys sorted by code point, no insignificant whitespace, UTF-8.
    NaN and Infinity are rejected rather than emitted as invalid JSON.
    """
    _check_finite(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    """`sha256:<hex>` over the canonical form of ``value``."""
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def request_id(state: Any, questions: dict[str, Any]) -> str:
    """Digest of state + questions, with no model.

    This is what pairs a golden-set record with its replay on a different
    model version, since the full record id includes the model and would
    therefore differ.
    """
    return digest({"questions": questions, "state": state})


def record_id(model: str, state: Any, questions: dict[str, Any]) -> str:
    """Digest of model + state + questions. The record's ``id`` field."""
    return digest({"model": model, "questions": questions, "state": state})
