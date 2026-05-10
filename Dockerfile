# Multi-stage: the build stage keeps compilers and wheels out of the runtime image.
# Dependencies come from pyproject.toml rather than a parallel requirements.txt, so
# there is exactly one place where a version is pinned.
FROM python:3.11-slim-bookworm AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /src
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --upgrade pip setuptools wheel && pip install .


FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8080

# A fixed high uid keeps the Kubernetes securityContext (runAsUser: 10001) and the
# image in agreement, and lets the pod run with a read-only root filesystem.
RUN groupadd --system --gid 10001 gateway \
    && useradd --system --uid 10001 --gid gateway --no-create-home gateway

WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --chown=gateway:gateway app ./app
COPY --chown=gateway:gateway config ./config

USER gateway
EXPOSE 8080

# No curl or wget in the slim image, and adding one for a probe would be a needless
# attack surface.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/health/live').read()"]

ENTRYPOINT ["python", "-m", "uvicorn", "app.main:create_app", "--factory"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
