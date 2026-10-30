FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# Capa de dependencias (se cachea si no cambia el lock)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

# Código, datos y entrenamiento (la puerta de calidad aborta el build si falla)
COPY src ./src
COPY app ./app
RUN uv sync --locked --no-dev \
    && uv run datos \
    && uv run entrenar

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 7860
CMD ["api"]
