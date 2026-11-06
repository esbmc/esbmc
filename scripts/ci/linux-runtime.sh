#!/usr/bin/env bash
# Runtime of the Linux regression on a runner that did not build ESBMC.
# Usage: linux-runtime.sh [all|packages|python]

set -euo pipefail

RUNTIME_PACKAGES=(
  python-is-python3
  python3-pip
  python3-toml
  z3            # z3_binary capability, bitwuzllob and neurosym suites
  default-jdk   # Jimple frontend
  g++-multilib  # --32 and host C++ headers
)

PYTHON_PACKAGES=(meson mypy pyparsing toml tomli pytest hypothesis)

sudo_if_needed() {
  if [[ "$(id -u)" -ne 0 ]]; then sudo "$@"; else "$@"; fi
}

install_packages() {
  # Best-effort update, as in build.sh: a third-party repository of the runner
  # image can serve a malformed index; a missing package still fails install.
  sudo_if_needed apt-get update || echo "warning: apt-get update failed; using cached indices"
  sudo_if_needed apt-get install -y --no-install-recommends "${RUNTIME_PACKAGES[@]}"
}

install_python() {
  python3 -m pip install --user --break-system-packages "${PYTHON_PACKAGES[@]}"
}

case "${1:-all}" in
  all) install_packages; install_python ;;
  packages) install_packages ;;
  python) install_python ;;
  *) echo "usage: $0 [all|packages|python]" >&2; exit 2 ;;
esac
