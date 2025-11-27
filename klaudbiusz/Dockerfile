FROM python:3.12-slim

# install databricks cli and node.js (for claude agent sdk)
RUN apt-get update && \
    apt-get install -y curl unzip && \
    curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @anthropic-ai/claude-code && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# install uv for package management
RUN pip install uv

# copy python dependencies and source code
COPY pyproject.toml README.md ./
COPY cli/ ./cli/

# install dependencies
RUN uv pip install --system -e .

# create non-root user (claude agent sdk requires non-root for security)
RUN useradd -m -s /bin/bash klaudbiusz && \
    chown -R klaudbiusz:klaudbiusz /workspace

USER klaudbiusz

# set working directory for app generation
ENV APP_OUTPUT_DIR=/workspace/app
