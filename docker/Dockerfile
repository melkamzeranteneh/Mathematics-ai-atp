# =============================================================================
# Mathematics AI ATP - Multi-stage Dockerfile
#
# Architecture:
#   - Shared base stage: elan/Lean/Mathlib cache, SWI-Prolog, petta, uv venv
#   - Three final stages: core, experimental, training
#
# Build targets:
#   docker build --target core -t maths_ai-core .
#   docker build --target experimental -t maths_ai-experimental .
#   docker build --target training -t maths_ai-training .
# =============================================================================

# ---------- base ----------
# Shared base stage with all slow dependencies (Lean/Mathlib, SWI-Prolog, petta)
FROM ubuntu:24.04 AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install elan (Lean version manager)
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
ENV PATH="/root/.elan/bin:${PATH}"

# Install uv for Python dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy lean_project files
COPY lean_project/ /opt/lean_project/
WORKDIR /opt/lean_project

# Build Lean project with Mathlib cache
# Uses BuildKit cache mount for .lake directory
RUN --mount=type=cache,target=/opt/lean_project/.lake \
    lake update && lake exe cache get && lake build

ENV MATHS_AI_LEAN_PROJECT=/opt/lean_project
ENV MATHS_AI_DATA_ROOT=/data

# Install SWI-Prolog >= 9.3.x (Ubuntu apt ships 9.0.4 which is too old)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    && apt-add-repository -y ppa:swi-prolog/stable \
    && apt-get update && apt-get install -y --no-install-recommends swi-prolog \
    && rm -rf /var/lib/apt/lists/*

# Install petta
ARG PETTA_REF=v1.0.3
RUN git clone --depth 1 --branch ${PETTA_REF} https://github.com/trueagi-io/PeTTa.git /opt/petta \
    && pip install /opt/petta

# Install PeTTaChainer (vendor at pinned commit, replace git-import in MeTTa)
ARG PETTACHAINER_REF=HEAD
RUN git clone https://github.com/rTreutlein/PeTTaChainer.git /opt/pettachainer \
    && git -C /opt/pettachainer checkout ${PETTACHAINER_REF}

# Set up workspace and install Python dependencies
WORKDIR /workspace
COPY pyproject.toml uv.lock /workspace/
RUN uv venv /opt/venv && \
    uv sync --no-dev && \
    uv pip install -e . --no-deps

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/workspace"

# Create non-root user for security
RUN useradd -m -u 1000 maths_ai

# Create data directory for assets
RUN mkdir -p /data/gnn_inference/runs

USER maths_ai
WORKDIR /workspace

# ---------- core ----------
# Core service: always running, serves proof/inference requests
FROM base AS core

# Copy application code
COPY maths_ai/ /workspace/maths_ai/
COPY scripts/ /workspace/scripts/
COPY docker/commands/ /workspace/docker/commands/
COPY docker/entrypoint.sh /entrypoint.sh

# Set permissions for entrypoint and command scripts
USER root
RUN chmod +x /entrypoint.sh && chmod +x /workspace/docker/commands/*.sh
USER maths_ai

# Health check: trivial proof/inference call
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m maths_ai.healthcheck || exit 1

ENTRYPOINT ["/entrypoint.sh"]

# ---------- experimental ----------
# Experimental service: sandbox for experiments/, on demand
FROM base AS experimental

# Copy application code including experiments
COPY maths_ai/ /workspace/maths_ai/
COPY scripts/ /workspace/scripts/
COPY docker/commands/ /workspace/docker/commands/
COPY docker/entrypoint.sh /entrypoint.sh
COPY experiments/ /workspace/experiments/

# Set permissions for entrypoint and command scripts
USER root
RUN chmod +x /entrypoint.sh && chmod +x /workspace/docker/commands/*.sh
USER maths_ai

ENTRYPOINT ["/entrypoint.sh"]

# ---------- training ----------
# Training service: batch job, GPU-optional, different lifecycle
FROM base AS training

# Copy application code
COPY maths_ai/ /workspace/maths_ai/
COPY scripts/ /workspace/scripts/
COPY docker/commands/ /workspace/docker/commands/
COPY docker/entrypoint.sh /entrypoint.sh

# Set permissions for entrypoint and command scripts
USER root
RUN chmod +x /entrypoint.sh && chmod +x /workspace/docker/commands/*.sh
USER maths_ai

ENTRYPOINT ["/entrypoint.sh"]
