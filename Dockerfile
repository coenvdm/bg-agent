# ── Base: PyTorch 2.3.1 + CUDA 12.1 (covers RTX 3000/4000 series on vast.ai) ──
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace/bg-dataset

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        tmux \
        vim \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (cached layer — only rebuilds if requirements.txt changes) ───
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .

# ── Checkpoints dir (vast.ai /workspace persists across restarts if mounted) ─
RUN mkdir -p /workspace/checkpoints

# ── Default: bash so you can run train.py / inspect logs interactively ────────
CMD ["/bin/bash"]
