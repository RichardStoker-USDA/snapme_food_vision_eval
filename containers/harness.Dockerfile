# Benchmark harness: ground truth builder, pipelines, scoring. CPU only; it talks to any
# OpenAI-compatible server, so the engine runs in its own container (see compose.yaml).
#   docker build -f containers/harness.Dockerfile -t snapme_eval-harness .
FROM ghcr.io/prefix-dev/pixi:0.84.1-noble@sha256:2d8821a1f3ee11a7fe8055f2c72d2b116d22f00c4caa73809857b5f216fcdebc AS build
WORKDIR /app
COPY pixi.toml pixi.lock pyproject.toml project.toml ./
COPY src ./src
RUN pixi install --locked -e default && \
    pixi shell-hook -e default -s bash > /shell-hook && \
    printf '#!/bin/bash\nsource /shell-hook\nexec "$@"\n' > /entrypoint.sh && chmod +x /entrypoint.sh

FROM ubuntu:24.04
WORKDIR /app
COPY --from=build /app /app
COPY --from=build /shell-hook /shell-hook
COPY --from=build /entrypoint.sh /entrypoint.sh
COPY configs ./configs
COPY tools ./tools
# Data, models and runs live outside the image. Extra model files go in /models_extra.
ENV SNAPME_ROOT=/data SNAPME_MODEL_DIRS=/models_extra NLTK_DATA=/data/env/nltk_data PYTHONNOUSERSITE=1
RUN mkdir -p /data /models_extra && chown ubuntu /data /models_extra   # ubuntu:24.04 ships user ubuntu (uid 1000)
USER ubuntu
ENTRYPOINT ["/entrypoint.sh", "snapme-eval"]
CMD ["--help"]
