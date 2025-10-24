# Dev/test container for GraphMER-SE
FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps (optional: git for versioning, curl for debugging)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip cache purge

# Copy source late to leverage layer caching for deps
COPY . /workspace

# Default command runs tests
CMD ["python", "-m", "pytest", "-q"]
