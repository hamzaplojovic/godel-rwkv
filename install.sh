#!/bin/bash
#
# install.sh — Install the GodelRWKV stuck-pattern supervisor for Claude Code.
#
# One command:
#   curl -sL https://raw.githubusercontent.com/hamzaplojovic/godel-rwkv/main/install.sh | bash
#
# What it does:
#   1. Clones the repo to ~/.godel-rwkv (or updates if exists)
#   2. Installs Python dependencies (mlx, numpy)
#   3. Adds the PostToolUse hook to ~/.claude/settings.json
#
# Requirements: python3, git, pip3
# Works on macOS (Apple Silicon for MLX)
#

set -e

INSTALL_DIR="$HOME/.godel-rwkv"
SETTINGS="$HOME/.claude/settings.json"
HOOK_CMD="python3 $INSTALL_DIR/main.py"

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

# 2. Install deps
echo "Installing dependencies..."
cd "$INSTALL_DIR"
pip3 install --quiet mlx numpy 2>/dev/null || pip install --quiet mlx numpy

# 3. Check model weights exist
if [ ! -f "$INSTALL_DIR/weights/multiclass.npz" ]; then
    echo "ERROR: Model weights not found at $INSTALL_DIR/weights/multiclass.npz"
    echo "Run: cd $INSTALL_DIR && uv run python training/train_multiclass.py"
    exit 1
fi

# 4. Add hook to settings.json
if [ ! -f "$SETTINGS" ]; then
    echo "ERROR: ~/.claude/settings.json not found. Is Claude Code installed?"
    exit 1
fi

# Check if hook already installed
if grep -q "godel-rwkv/main.py" "$SETTINGS" 2>/dev/null; then
    echo "Hook already installed in $SETTINGS"
else
    echo "Adding PostToolUse hook to $SETTINGS..."
    # Use python3 to safely modify JSON
    python3 -c "
import json
from pathlib import Path

settings_path = Path('$SETTINGS')
settings = json.loads(settings_path.read_text())

hook_entry = {
    'matcher': '',
    'hooks': [{
        'type': 'command',
        'command': '$HOOK_CMD',
        'timeout': 10
    }]
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
echo "The supervisor is now active. It will:"
echo "  • Watch every Claude Code tool call (5ms overhead)"
echo "  • Stay silent while you're making progress"
echo "  • Alert when it detects a stuck pattern (edit-revert, read-cycle, test-fail loop)"
echo "  • Show git context and suggest what file to look at instead"
echo ""
echo "To uninstall: remove the godel-rwkv PostToolUse entry from ~/.claude/settings.json"
echo "              and rm -rf ~/.godel-rwkv"
