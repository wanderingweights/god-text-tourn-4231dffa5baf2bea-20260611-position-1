FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    vim \
    zip \
    tmux \
    iotop \
    nvtop \
    bmon \
    wget \
    nano \
    zsh \
    htop \
    && rm -rf /var/lib/apt/lists/*
# Default dir
RUN mkdir -p /workspace
RUN mkdir -p /cache
RUN mkdir -p /workspace/scripts/datasets
RUN mkdir -p /app/checkpoints
WORKDIR /workspace/scripts
# Copy current folder to /workspace/auto_ml
COPY scripts /workspace/scripts
# Make entrypoint script executable
RUN chmod +x /workspace/scripts/entrypoint.sh
# Pytorch (Auto-selects backend https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)

# Create a virtual environment for data generation
RUN python -m venv /workspace/axo_py
RUN bash -c "source /workspace/axo_py/bin/activate && \
    pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    uv pip install --no-build-isolation axolotl==0.9.1 && \
    pip install requests==2.32.3 && \
    deactivate"


# install the main dependencies
# NOTE: bumped to torch 2.7.1 / transformers 5.12.1 / triton 3.3.1 to support
# custom hybrid-attention checkpoints (e.g. silx-ai/Quasar-Preview), whose
# vendored fla kernels require torch>=2.5 (torch.distributed.tensor public API)
# and whose modeling code is authored against transformers v5.
# torch is installed FIRST so flash-attn compiles against 2.7.
RUN pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 && \
    uv pip install -r /workspace/scripts/training_requirements.txt --system && \
    pip install hf_transfer==0.1.9 && \
    pip install tenacity==9.1.2 && \
    pip install tiktoken==0.9.0 && \
    pip install flash-attn==v2.7.4.post1 --no-build-isolation && \
    pip install "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0"

# The runpod base image ships torch-2.4 torchvision/torchaudio; the torch 2.7.1
# bump above leaves them ABI-mismatched (transformers lazily imports them ->
# "operator torchvision::nms does not exist" / libtorchaudio undefined symbol).
# The text trainer needs neither — remove them so those imports are skipped.
RUN pip uninstall -y torchvision torchaudio || true
# vLLM (GRPO rollouts only) is intentionally NOT installed here: vllm==0.8.3
# pins torch 2.4 and conflicts with the torch 2.7 bump above. GRPO support on
# the bumped stack needs a torch-2.7-compatible vllm (>=0.10) and is a separate
# follow-up; the instruct/DPO training paths do not import vllm.

ENTRYPOINT ["./entrypoint.sh"]