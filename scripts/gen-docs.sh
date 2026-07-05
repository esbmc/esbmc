#!/usr/bin/env bash
#
# Generate the ESBMC Doxygen API reference from the single .doxygen config.
#
# Usage: scripts/build-docs.sh [OUTPUT_DIRECTORY [HTML_OUTPUT]]
#
# With no arguments the HTML goes to docs/html (the Doxyfile default) — used by
# the local build, the CMake `docs` target, and the CI check. The website
# deploy passes an output directory and HTML subdir to publish under /docs/api.
# PROJECT_NUMBER is stamped from CMakeLists.txt here so the version lives in
# exactly one place rather than being re-parsed by every consumer.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

version=$(sed -nE 's/^set\(ESBMC_VERSION_(MAJOR|MINOR|PATCH) ([0-9]+)\).*/\2/p' \
  CMakeLists.txt | paste -sd.)

if [ $# -ge 1 ]; then mkdir -p "$1"; fi

{
  cat .doxygen
  echo "PROJECT_NUMBER = $version"
  if [ $# -ge 1 ]; then echo "OUTPUT_DIRECTORY = $1"; fi
  if [ $# -ge 2 ]; then echo "HTML_OUTPUT = $2"; fi
} | doxygen -
