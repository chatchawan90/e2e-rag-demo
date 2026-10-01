FROM python:3.12-slim
LABEL org.opencontainers.image.authors="Chatchawan Lakkhananukun" \
      org.opencontainers.image.vendor="Chatchawan Lakkhananukun" \
      org.opencontainers.image.title="Envsearch" \
      org.opencontainers.image.source="https://github.com/chatchawan90/e2e-rag-demo"
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    ENVSEARCH_PUBLIC_DEMO=1 ENVSEARCH_EMBEDDING_BACKEND=onnx \
    ENVSEARCH_HOME=/app/deploy/demo-data HF_HOME=/opt/models \
    HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false \
    ENVSEARCH_ONNX_MODEL=/opt/models/e5/model.onnx ENVSEARCH_ONNX_MAX_TOKENS=256
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends nginx libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md COPYRIGHT ./
COPY src ./src
COPY corpus ./corpus
COPY evals/questions.yaml ./evals/questions.yaml
COPY deploy ./deploy
RUN pip install -e '.[web,hosted]' 'onnx>=1.17,<2' \
    && python deploy/prepare_model.py \
    && useradd --uid 1000 --create-home demo \
    && chown -R demo:demo /app /opt/models
USER demo
ENV HF_HUB_OFFLINE=1
EXPOSE 10000
CMD ["python", "-m", "envsearch.hosted"]
