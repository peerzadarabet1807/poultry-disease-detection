# Poultry disease classifier API (CPU). Build from the repository root:
#   docker build -f backend/Dockerfile -t poultry-disease-api .
# The weights are not in the image by default: on first start they are downloaded from the GitHub Release
# into the cache volume and checked against the SHA-256 in the package. Pass --build-arg BAKE_WEIGHTS=1 to
# download and verify them at build time instead (a self-contained image, about 95 MB larger).

FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    POULTRY_API_REPO_ROOT=/srv \
    POULTRY_DISEASE_CACHE=/home/app/.cache/poultry-disease

WORKDIR /srv

# Dependencies first, so code changes do not reinstall TensorFlow.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[api]" && rm -rf src build

RUN useradd --create-home --uid 10001 app && mkdir -p "$POULTRY_DISEASE_CACHE" && chown -R app:app /home/app

# Files GET /model reads (metrics, intervals, calibration, config, checksums) and the app itself.
COPY models/model_config.json models/checksums.sha256 ./models/
COPY results/experiment_config.json ./results/
COPY results/posthoc/bootstrap_ci.json results/posthoc/calibration.json ./results/posthoc/
COPY backend/poultry_api ./poultry_api

USER app

ARG BAKE_WEIGHTS=0
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then python -c "from poultry_disease.weights import get_weights; print(get_weights())"; fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"

# One worker: the model is loaded once per process and TensorFlow already uses every core per request.
CMD ["uvicorn", "poultry_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
