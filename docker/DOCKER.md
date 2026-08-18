# Mathematics AI ATP - Docker Documentation

## Overview

This project uses a multi-stage Dockerfile with a shared base stage and three service-specific targets:

- **base** — Shared dependencies: Lean 4.15.0 + Mathlib4, SWI-Prolog 9.3+, petta, uv venv
- **core** — Core service: always-running proof/inference server
- **experimental** — Experimental service: sandbox for experiments on demand
- **training** — Training service: batch job, GPU-optional, different lifecycle

## Quick Start

### Build Images

```bash
# Build core service (production proof/inference server)
docker build --target core -t maths_ai-core -f docker/Dockerfile .

# Build experimental service (includes experiments/)
docker build --target experimental -t maths_ai-experimental -f docker/Dockerfile .

# Build training service (batch jobs, GPU-optional)
docker build --target training -t maths_ai-training -f docker/Dockerfile .
```

### Run Services

```bash
# Core service (health-checked, always running)
docker run -d --name maths_ai-core \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  maths_ai-core

# Experimental service (interactive)
docker run -it --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/experiments:/workspace/experiments \
  maths_ai-experimental

# Training service (batch, with GPU support)
docker run --rm --gpus all \
  -v $(pwd)/data:/data \
  maths_ai-training python -m maths_ai.gnn_inference.train
```

## Architecture

### Multi-Stage Design

```
┌─────────────────────────────────────────────────────────────┐
│                        BASE STAGE                           │
│  Ubuntu 24.04                                               │
│  ├── System deps (ca-certificates, curl, git, python3)      │
│  ├── elan → Lean 4.15.0                                     │
│  ├── Mathlib4 (cached via lake)                             │
│  ├── uv (Python package manager)                            │
│  ├── SWI-Prolog 9.3+ (from PPA)                             │
│  ├── petta (v1.0.3)                                         │
│  └── PeTTaChainer (pinned commit)                           │
│  └── uv venv + Python deps (pyproject.toml)                 │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌─────────┐    ┌──────────────┐ ┌──────────┐
         │  CORE   │    │ EXPERIMENTAL │ │ TRAINING │
         │         │    │              │ │          │
         │ maths_ai│    │ maths_ai +   │ │ maths_ai │
         │ scripts │    │ experiments  │ │ scripts  │
         │ entrypt │    │ entrypt      │ │ entrypt  │
         │ health  │    │              │ │          │
         └─────────┘    └──────────────┘ └──────────┘
```

### Cache Strategy

- **BuildKit cache mount** for `/opt/lean_project/.lake` — persists Mathlib build artifacts across builds
- **Layer caching** — base stage cached unless system deps or Lean toolchain changes
- **uv cache** — Python dependencies cached in `/root/.cache/uv`

## Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `PETTA_REF` | `v1.0.3` | petta git tag/branch |
| `PETTACHAINER_REF` | `HEAD` | PeTTaChainer commit/ref |

```bash
docker build --target core \
  --build-arg PETTA_REF=v1.0.3 \
  --build-arg PETTACHAINER_REF=abc123 \
  -t maths_ai-core -f docker/Dockerfile .
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATHS_AI_LEAN_PROJECT` | `/opt/lean_project` | Lean project root |
| `MATHS_AI_DATA_ROOT` | `/data` | Data directory for assets |
| `PATH` | `/opt/venv/bin:...` | Includes venv, elan, uv |
| `PYTHONPATH` | `/workspace` | Python module path |

## Volumes

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/data` | Persistent data (GNN runs, datasets) |
| `./experiments` | `/workspace/experiments` | Experimental code (experimental target only) |

## Entrypoint

All targets use `/entrypoint.sh` which:
1. Sets up environment
2. Executes passed command or defaults to service-specific entry

```bash
# Override entrypoint
docker run maths_ai-core python -m maths_ai.hybrid_reasoner.repl
```

## Health Check (core target)

```bash
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -m maths_ai.healthcheck || exit 1
```

## GPU Support (training target)

```bash
# NVIDIA GPU
docker run --rm --gpus all maths_ai-training python -m maths_ai.gnn_inference.train

# Specific GPU
docker run --rm --gpus '"device=0,1"' maths_ai-training ...
```

## Troubleshooting

### Build Failures

**Mathlib clone timeout (exit code 128)**
- The Dockerfile includes git config for longer timeouts
- If persistent, try: `docker build --no-cache --target core ...`

**DNS resolution errors**
- Base stage configures Google (8.8.8.8) and Cloudflare (1.1.1.1) DNS
- On corporate networks, may need additional proxy config

**Lean/Mathlib version mismatch**
- `lean-toolchain` in `lean_project/` pins Lean version
- `lakefile.toml` pins Mathlib rev (`v4.15.0`)
- Update both together

### Slow Builds

- First build downloads ~2GB (Lean, Mathlib, Python deps)
- Subsequent builds use BuildKit cache for `.lake` directory
- Run with `DOCKER_BUILDKIT=1` (default on Docker Desktop)

### Runtime Issues

**Permission denied on /data**
```bash
# Fix host permissions
sudo chown -R 1000:1000 ./data
```

**Module not found**
- Ensure `PYTHONPATH=/workspace` is set
- Check `uv sync` completed in base stage

## CI/CD Integration

```yaml
# Example GitHub Actions
- name: Build core image
  run: |
    docker build --target core -t maths_ai-core -f docker/Dockerfile .
    docker run --rm maths_ai-core python -m pytest tests/
```

## Development Workflow

```bash
# Rebuild after Python code changes (fast - uses cached base)
docker build --target core -t maths_ai-core -f docker/Dockerfile .

# Rebuild after lean_project changes (rebuilds lake)
docker build --target core --no-cache -t maths_ai-core -f docker/Dockerfile .

# Interactive development
docker run -it --rm -v $(pwd):/workspace maths_ai-core bash
```

## Security

- Non-root user `maths_ai` (UID 1000) runs services
- Base image: Ubuntu 24.04 LTS (security updates)
- No secrets in image — use runtime env vars / Docker secrets
- Minimal attack surface: no build tools in final stages