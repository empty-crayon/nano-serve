set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODAL_DIR="$CLASS_ROOT/modal"
cd "$CLASS_ROOT"

if ! command -v modal &>/dev/null; then
  echo "ERROR: Modal is required. Install: pip install modal && modal token set"
  exit 1
fi

echo "Deploying Modal app (relay-llama-server)..."
cd "$MODAL_DIR"
modal deploy modal_llama_server.py

echo ""
echo "Fetching web URL..."
# modal app show or similar to get URL; deploy typically prints it
# If deploy doesn't print URL, user can run: modal run modal_llama_server.py --serve
# and get the URL from the output. For deploy we get the URL from the app.
URL=$(modal app list 2>/dev/null | grep -E "relay-llama-server|web" | head -1 || true)
if [ -z "$URL" ]; then
  echo "To get your public URL, run:"
  echo "  cd $MODAL_DIR && modal run modal_llama_server.py"
  echo "  Then copy the 'https://...modal.run' URL from the output."
fi
echo ""
echo "Required next steps:"
echo "  1. Copy the Modal web URL (from 'modal run modal_llama_server.py' or the deploy output)."
echo "  2. Edit RelayServe/config.yaml and set backends.modal.url to that URL (no trailing slash)."
echo "     Example: url: https://your-workspace--relay-llama-server-web.modal.run"
echo "  3. Start local llama (scripts/spawn_backends.py), then RelayServe (scripts/start_relay_tinyllama.sh)."
echo "  4. Run: bash scripts/test_modal_backend.sh  (tests both local and Modal)."
