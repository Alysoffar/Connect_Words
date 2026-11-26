"""Structured API errors shared by routes and exception handlers."""

from __future__ import annotations

from typing import Any


class APIError(Exception):
    """An expected API failure with a stable machine-readable code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}
