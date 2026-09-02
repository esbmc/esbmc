#!/usr/bin/env bash
# PostToolUse hook: clang-format the lines Claude Code just touched.
# Reads the hook JSON payload from stdin, per Claude Code's hook contract.
set -uo pipefail

if ! command -v jq >/dev/null 2>&1; then
  echo "clang-format-fix: jq not found on PATH; skipping (install jq to enable auto-formatting)." >&2
  exit 0
fi

file=$(jq -r '.tool_input.file_path // empty')

case "$file" in
  *.c | *.cpp | *.h | *.hpp | *.hh) ;;
  *) exit 0 ;;
esac

[ -f "$file" ] || exit 0

# Prefer the clang-format 11 / git-clang-format pair pinned by
# scripts/install-format-hook.sh, since that's the exact version CI's
# code-style check uses, and git-clang-format lets us format only the lines
# Claude just touched instead of rewriting the whole file (matching CI's and
# install-format-hook.sh's changed-lines-only scope).
cache_dir="$CLAUDE_PROJECT_DIR/.cache/clang-format-11/bin"
cf="$cache_dir/clang-format"
gcf="$cache_dir/git-clang-format"

[ -x "$cf" ] || cf=$(command -v clang-format || true)
if [ -z "$cf" ] || ! "$cf" --version 2>/dev/null | grep -q 'version 11\.'; then
  echo "clang-format-fix: no clang-format 11 available; run scripts/install-format-hook.sh to install the pinned version." >&2
  exit 0
fi

[ -x "$gcf" ] || gcf=$(command -v git-clang-format || true)
if [ -z "$gcf" ]; then
  echo "clang-format-fix: git-clang-format not available; run scripts/install-format-hook.sh to install it." >&2
  exit 0
fi

# git-clang-format diffs against git history, so it never sees a file git
# doesn't know about yet: format brand-new (untracked) files whole instead —
# there's no pre-existing history to accidentally sweep up.
if ! git -C "$CLAUDE_PROJECT_DIR" ls-files --error-unmatch "$file" >/dev/null 2>&1; then
  "$cf" -i "$file"
  exit 0
fi

base=$(git -C "$CLAUDE_PROJECT_DIR" merge-base HEAD origin/master 2>/dev/null || echo origin/master)
( cd "$CLAUDE_PROJECT_DIR" && python3 "$gcf" --force --binary "$cf" --style file \
    --extensions h,c,cpp,hpp,hh "$base" -- "$file" >/dev/null 2>&1 )

exit 0
