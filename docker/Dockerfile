# ==========================================
# ⬇️ PHASE 0: JULIA SOURCE ⬇️
# ==========================================
FROM julia:1.11 AS julia-base

# ==========================================
# ⬇️ PHASE 1: ROOT SETUP ⬇️
# ==========================================
FROM condaforge/miniforge3
LABEL maintainer="ANONYMIZED"

# 1. Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo curl bzip2 tree neovim build-essential wget gnupg ca-certificates \
    && wget -q -O /tmp/google-chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get install -y /tmp/google-chrome.deb \
    && rm /tmp/google-chrome.deb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Copy Julia from the official image
COPY --from=julia-base /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:${PATH}"

# 3. Clear UID 1000 and setup vscode user
RUN if getent passwd 1000; then userdel -r $(getent passwd 1000 | cut -d: -f1); fi && \
    useradd -m -u 1000 -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-vscode && \
    chown -R vscode:vscode /opt/conda && \
    mkdir -p /opt/julia_depot && chown -R vscode:vscode /opt/julia_depot

ENV JULIA_DEPOT_PATH="/opt/julia_depot"

# ==========================================
# ⬇️ PHASE 2: VSCODE USER SETUP ⬇️
# ==========================================
USER vscode
WORKDIR /home/vscode

# 4. Build the environment using Mamba
# v3.0: minimal spec, let conda resolve latest compatible versions
COPY --chown=vscode:vscode ./src/environment.yaml /tmp/docker/src/environment.yaml
RUN mamba env create -f /tmp/docker/src/environment.yaml --yes && \
    mamba clean -afy

# 5. Pre-compile Julia dependencies
# Importing juliacall + calling using PythonCall triggers juliapkg to install,
# resolve, and JIT-compile PythonCall. JULIAPKG_OFFLINE=yes (set below) then
# suppresses all setup messages on subsequent runs.
RUN /opt/conda/envs/boolean_reservoir/bin/python -c "\
from juliacall import Main as jl; \
jl.seval('using PythonCall'); \
jl.println('Julia + PythonCall precompiled and ready')" && \
    julia -e 'using Pkg; Pkg.gc()' && \
    rm -f /opt/julia_depot/environments/pyjuliapkg/lock.pid

# 6. Final Environment Variables
ENV PYTHONPATH="/code"
ENV JULIAPKG_OFFLINE="yes"

# Make the boolean_reservoir conda env the default `python`/`python3` on PATH for
# every process (extension hosts, tasks, non-login shells), not just interactive
# bash shells that source ~/.bashrc.
ENV CONDA_DEFAULT_ENV=boolean_reservoir
ENV CONDA_PREFIX="/opt/conda/envs/boolean_reservoir"
ENV PATH="/opt/conda/envs/boolean_reservoir/bin:${PATH}"

RUN conda init bash && echo 'conda activate boolean_reservoir' >> ~/.bashrc

WORKDIR /code
CMD ["bash"]
