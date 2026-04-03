FROM python:3.11-slim

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY app ./app
COPY evals ./evals
COPY docs ./docs
COPY data ./data

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
