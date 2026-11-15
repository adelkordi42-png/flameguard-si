# Digest-pinned for reproducible builds (linux/amd64). Refresh with:
#   docker buildx imagetools inspect python:3.14-slim
FROM python:3.14-slim@sha256:ac25e5afbfdb4a94ff99cc24b0345719210133144481480bcba6085efa36a341

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/user

RUN useradd -m -u 1000 user

WORKDIR /app
RUN chown user /app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the committed data, schema, combiner, metrics and Jev cache. There is no
# training step at build time: the live service must serve exactly the artifacts
# that VALIDATION.md describes.
COPY --chown=user . /app

USER user

EXPOSE 7860

CMD ["sh", "-c", "exec uvicorn flameguard.api:app --host 0.0.0.0 --port ${PORT:-7860}"]