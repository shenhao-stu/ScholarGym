#!/usr/bin/env bash
# start_services.sh — Start Qdrant (via docker-compose) and Ollama, with health checks.
#
# Usage:
#   ./scripts/start_services.sh                 # Start services only
#   ./scripts/start_services.sh --build-index   # Start services + build Qdrant vector index
#
# Environment variables (override config.py defaults):
#   QDRANT_URL          default: http://localhost:6433
#   OLLAMA_URL          default: http://localhost:11434
#   EMBEDDING_MODEL     default: qwen3-embedding:0.6b

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

QDRANT_URL="${QDRANT_URL:-http://localhost:6433}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding:0.6b}"

HEALTH_TIMEOUT=60        # seconds to wait for each service
HEALTH_INTERVAL=2         # seconds between retries
BUILD_INDEX=false

# Bypass proxy for local services (common in lab/campus environments)
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --build-index) BUILD_INDEX=true ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { log "OK  $*"; }
fail() { log "FAIL $*"; exit 1; }

wait_for_url() {
    local url="$1" name="$2"
    local elapsed=0
    log "Waiting for $name at $url ..."
    while ! curl -sf --max-time 3 "$url" > /dev/null 2>&1; do
        sleep "$HEALTH_INTERVAL"
        elapsed=$((elapsed + HEALTH_INTERVAL))
        if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
            fail "$name did not become healthy within ${HEALTH_TIMEOUT}s"
        fi
    done
    ok "$name is healthy (${elapsed}s)"
}

# ── 1. Start Qdrant via docker-compose ────────────────────────────────────────
log "Starting Qdrant via docker compose ..."
cd "$PROJECT_ROOT"

if ! command -v docker &> /dev/null; then
    fail "docker is not installed. Please install Docker first."
fi

docker compose up -d
ok "docker compose up -d"

# Health check — Qdrant REST root returns {"title":"qdrant - vectorass engine",...}
wait_for_url "$QDRANT_URL" "Qdrant"

# ── 2. Start / check Ollama ──────────────────────────────────────────────────
log "Checking Ollama ..."

if curl -sf --max-time 3 "$OLLAMA_URL" > /dev/null 2>&1; then
    ok "Ollama is already running"
else
    if command -v ollama &> /dev/null; then
        log "Starting Ollama server ..."
        ollama serve > /dev/null 2>&1 &
        wait_for_url "$OLLAMA_URL" "Ollama"
    else
        fail "Ollama is not installed and not reachable at $OLLAMA_URL. Please install Ollama: https://ollama.com"
    fi
fi

# Ensure the embedding model is available
log "Ensuring embedding model '$EMBEDDING_MODEL' is available ..."
if ! ollama list 2>/dev/null | grep -q "$EMBEDDING_MODEL"; then
    log "Pulling $EMBEDDING_MODEL (this may take a while) ..."
    ollama pull "$EMBEDDING_MODEL"
fi
ok "Embedding model ready"

# ── 3. Optional: build vector index ──────────────────────────────────────────
if [ "$BUILD_INDEX" = true ]; then
    log "Building Qdrant vector index ..."
    python "$PROJECT_ROOT/code/build_vector_db.py" \
        --paper_db "$PROJECT_ROOT/data/scholargym_paper_db.json" \
        --qdrant_url "$QDRANT_URL" \
        --ollama_url "$OLLAMA_URL" \
        --embedding_model "$EMBEDDING_MODEL"
    ok "Vector index built"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
log "All services are up."
log "  Qdrant:  $QDRANT_URL"
log "  Ollama:  $OLLAMA_URL"
echo ""
log "Run evaluation with:"
log "  python code/eval.py --search_method bm25"
log "  python code/eval.py --search_method vector"
