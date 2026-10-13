#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=pd-model-api

if [ ! -f models/model.joblib ]; then
    echo "models/model.joblib not found, running dvc repro"
    dvc repro
fi

docker build -t "$IMAGE" .
docker stop "$IMAGE" >/dev/null 2>&1 || true
docker run -d --rm --name "$IMAGE" -p 8000:8000 "$IMAGE"

for _ in $(seq 1 30); do
    curl -sf localhost:8000/health >/dev/null && break
    sleep 1
done

echo "POST /predict:"
curl -s -X POST localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d @scripts/example.json
echo
