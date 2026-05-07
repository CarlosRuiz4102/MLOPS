FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uv", "run", "uvicorn", "src.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
