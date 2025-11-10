#!/bin/bash
set -e

# Load .env file if it exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
fi

# Check required env vars
if [ -z "$DATABRICKS_HOST" ] || [ -z "$DATABRICKS_TOKEN" ]; then
    echo "❌ Error: DATABRICKS_HOST and DATABRICKS_TOKEN must be set"
    exit 1
fi

# Detect app structure and start
if [ -f "Dockerfile" ]; then
    echo "❌ Error: Docker apps not supported by start.sh"
    echo "   Use 'docker build' and 'docker run' instead"
    exit 1

elif [ -d "server" ] && [ -f "server/package.json" ]; then
    # tRPC style: server/ directory
    cd server && npm start

elif [ -f "package.json" ]; then
    # DBX SDK style: root package.json (with backend/)
    npm start

else
    echo "❌ Error: No package.json found"
    exit 1
fi
