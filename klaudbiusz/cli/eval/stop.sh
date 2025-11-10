#!/bin/bash

# Docker-based apps
if [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then
    docker compose down 2>/dev/null || true

elif [ -f "Dockerfile" ]; then
    # Stop container with app name
    APP_NAME=$(basename "$PWD")
    docker stop "$APP_NAME" 2>/dev/null || true
    docker rm "$APP_NAME" 2>/dev/null || true

    # Stop any eval-* or test-* containers
    for container in $(docker ps -aq --filter "name=eval-${APP_NAME}" --filter "name=test-${APP_NAME}" 2>/dev/null); do
        docker stop "$container" 2>/dev/null || true
        docker rm "$container" 2>/dev/null || true
    done
fi

# npm-based apps: kill port 8000 and npm processes
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
pkill -f "npm start" 2>/dev/null || true
pkill -f "tsx backend/index.ts" 2>/dev/null || true
pkill -f "tsx server/index.ts" 2>/dev/null || true
