#!/bin/bash
set -e

echo "🛑 Stopping all Databricks apps..."

# Stop Docker containers first (eval-* and test-*)
echo "  Checking for Docker containers..."
EVAL_CONTAINERS=$(docker ps -aq --filter "name=eval-" --filter "name=test-" 2>/dev/null || true)
if [ -n "$EVAL_CONTAINERS" ]; then
    echo "  Stopping Docker containers..."
    echo "$EVAL_CONTAINERS" | xargs docker stop 2>/dev/null || true
    echo "  Removing Docker containers..."
    echo "$EVAL_CONTAINERS" | xargs docker rm 2>/dev/null || true
    echo "  ✓ Docker containers cleaned up"
else
    echo "  ✓ No Docker containers to clean up"
fi

# Kill processes on port 8000
if lsof -ti:8000 >/dev/null 2>&1; then
    echo "  Killing processes on port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    echo "  ✓ Port 8000 freed"
else
    echo "  ✓ No processes on port 8000"
fi

# Kill any npm/node processes running from app directories
echo "  Cleaning up app processes..."
pkill -f "npm start" 2>/dev/null || true
pkill -f "tsx backend/index.ts" 2>/dev/null || true
pkill -f "tsx server/index.ts" 2>/dev/null || true

# Wait for processes to terminate
sleep 1

# Verify port is free
if lsof -ti:8000 >/dev/null 2>&1; then
    echo "  ⚠️  Warning: Port 8000 still in use"
    lsof -ti:8000
    exit 1
else
    echo "  ✓ All apps stopped successfully"
fi
