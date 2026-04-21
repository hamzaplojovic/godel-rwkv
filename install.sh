#!/bin/bash
#
# install.sh — Install the GodelRWKV stuck-pattern supervisor for Claude Code.
#
# One command:
#   curl -sL https://raw.githubusercontent.com/hamzaplojovic/godel-rwkv/main/install.sh | bash
#
# What it does:
#   1. Clones the repo to ~/.godel-rwkv (or updates if exists)
#   2. Exports weights to flat binary (weights/*.bin)
#   3. Compiles the C daemon (bin/godelrd) — no Python at inference time
#   4. Installs the launchd plist so the daemon starts on login
#   5. Installs Python deps (mlx, numpy) as fallback when daemon is down
#   6. Adds the PostToolUse hook to ~/.claude/settings.json
#
# Requirements: git, cc (Xcode CLT), python3
# macOS only (Apple Silicon for MLX fallback)
#

set -e

INSTALL_DIR="$HOME/.godel-rwkv"
SETTINGS="$HOME/.claude/settings.json"
HOOK_CMD="python3 $INSTALL_DIR/main.py"
DAEMON_BIN="$INSTALL_DIR/bin/godelrd"
PLIST_LABEL="com.hamzaplojovic.godelrd"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"

echo "=== GodelRWKV Supervisor Install ==="
echo ""

# 1. Clone or update
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating $INSTALL_DIR..."
    cd "$INSTALL_DIR" && git pull --quiet
else
    echo "Cloning to $INSTALL_DIR..."
    git clone --quiet https://github.com/hamzaplojovic/godel-rwkv.git "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# 2. Check weights exist
if [ ! -f "weights/classifier.npz" ] || [ ! -f "weights/success.npz" ]; then
    echo "ERROR: Model weights not found. Run training first:"
    echo "  cd $INSTALL_DIR"
    echo "  uv run training/train_classifier.py"
    echo "  uv run training/train_success.py"
    exit 1
fi

# 3. Export .npz → .bin (only if bin is older than npz)
if [ ! -f "weights/classifier.bin" ] || [ "weights/classifier.npz" -nt "weights/classifier.bin" ]; then
    echo "Exporting weights to binary..."
    python3 tools/export_weights.py
fi

# 4. Compile C daemon
if [ ! -f "$DAEMON_BIN" ] || [ "daemon/godel.c" -nt "$DAEMON_BIN" ] || [ "daemon/daemon.c" -nt "$DAEMON_BIN" ]; then
    echo "Compiling daemon..."
    make -C daemon
fi

# 5. Install Python deps (MLX fallback)
echo "Installing Python dependencies (MLX fallback)..."
pip3 install --quiet mlx numpy 2>/dev/null || pip install --quiet mlx numpy

# 6. Install launchd plist (start daemon on login)
echo "Installing launchd agent..."
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${DAEMON_BIN}</string>
        <string>${INSTALL_DIR}/weights</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>${HOME}/.cache/godel-rwkv/daemon.log</string>
</dict>
</plist>
PLIST

mkdir -p "$HOME/.cache/godel-rwkv"
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"
echo "  Daemon started (socket: /tmp/godel.sock)"

# 7. Add hook to settings.json
if [ ! -f "$SETTINGS" ]; then
    echo "ERROR: ~/.claude/settings.json not found. Is Claude Code installed?"
    exit 1
fi

if grep -q "godel-rwkv/main.py" "$SETTINGS" 2>/dev/null; then
    echo "Hook already installed in $SETTINGS"
else
    echo "Adding PostToolUse hook to $SETTINGS..."
    python3 -c "
import json
from pathlib import Path

settings_path = Path('$SETTINGS')
settings = json.loads(settings_path.read_text())

hook_entry = {
    'matcher': '',
    'hooks': [{'type': 'command', 'command': '$HOOK_CMD', 'timeout': 10}]
}

if 'hooks' not in settings:
    settings['hooks'] = {}
if 'PostToolUse' not in settings['hooks']:
    settings['hooks']['PostToolUse'] = []

settings['hooks']['PostToolUse'].append(hook_entry)
settings_path.write_text(json.dumps(settings, indent=2))
print('  Done.')
"
fi

echo ""
echo "=== Installed ==="
echo ""
echo "  Daemon:   $DAEMON_BIN  (~3MB RAM, <1ms inference)"
echo "  Fallback: Python + MLX (auto-used if daemon is down)"
echo "  Traces:   ~/.cache/godel-rwkv/traces.jsonl"
echo ""
echo "To uninstall:"
echo "  launchctl unload $PLIST_PATH && rm $PLIST_PATH"
echo "  Remove godel-rwkv PostToolUse entry from ~/.claude/settings.json"
echo "  rm -rf ~/.godel-rwkv"
