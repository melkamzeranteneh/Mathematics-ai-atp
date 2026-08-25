"""Run the maths_ai API service: ``python -m maths_ai.api``.

Host/port are configurable via ``MATHS_AI_API_HOST`` / ``MATHS_AI_API_PORT``.
"""

import os

import uvicorn

from maths_ai.api.app import create_app


def main() -> None:
    app = create_app()
    uvicorn.run(
        app,
        host=os.getenv("MATHS_AI_API_HOST", "0.0.0.0"),
        port=int(os.getenv("MATHS_AI_API_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
