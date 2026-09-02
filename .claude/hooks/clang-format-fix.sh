#!/usr/bin/env bash
# PostToolUse hook: clang-format the file Claude Code just wrote/edited.
# Reads the hook JSON payload from stdin, per Claude Code's hook contract.
set -uo pipefail

file=$(jq -r '.tool_input.file_path // empty' 2>/dev/null || true)

case "$file" in
  *.c | *.cpp | *.h | *.hpp | *.hh) ;;
  *) exit 0 ;;
esac

[ -f "$file" ] || exit 0

# Prefer the clang-format 11 pinned by scripts/install-format-hook.sh, since
# that's what CI's code-style check uses; fall back to PATH otherwise.
cf="$CLAUDE_PROJECT_DIR/.cache/clang-format-11/bin/clang-format"
if [ ! -x "$cf" ]; then
  cf=$(command -v clang-format || true)
fi

[ -n "$cf" ] && "$cf" -i "$file"

exit 0
