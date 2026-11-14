# Digest-pinned for reproducible builds (linux/amd64). Refresh with:
#   docker buildx imagetools inspect python:3.14-slim
FROM python:3.14-slim@sha256:ac25e5afbfdb4a94ff99cc24b0345719210133144481480bcba6085efa36a341

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/user

# Hugging Face Spaces (and other PaaS hosts) run the container as an
# unprivileged user, so UID 1000 must exist and own the working directory
# before artifacts are written into it.
RUN useradd -m -u 1000 user

WORKDIR /app
RUN chown user /app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user flameguard/ ./flameguard/
COPY --chown=user static/ ./static/
COPY --chown=user data/ ./data/

USER user

# Train inside the image so the served artifact always matches this source
# revision. Deterministic: the seed is pinned in flameguard/train.py.
RUN python -m flameguard.train

EXPOSE 7860

# $PORT is injected by Render/Heroku-style hosts; 7860 is the Hugging Face
# Spaces default from the README frontmatter. exec keeps PID 1 on uvicorn so
# it receives SIGTERM and shuts down cleanly.
CMD ["sh", "-c", "exec uvicorn flameguard.api:app --host 0.0.0.0 --port ${PORT:-7860}"]