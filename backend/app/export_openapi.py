"""Export the FastAPI-generated contract to backend/openapi.json."""

from __future__ import annotations

import json
from pathlib import Path

from .main import create_app


def main() -> None:
    """Write the generated OpenAPI document next to the backend package."""
    output_path = Path(__file__).resolve().parents[1] / "openapi.json"
    output_path.write_text(
        json.dumps(create_app().openapi(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
