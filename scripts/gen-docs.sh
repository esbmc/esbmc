#!/usr/bin/env bash
#
# Generate the ESBMC Doxygen API reference from the single .config/Doxyfile config.
#
# Usage: scripts/gen-docs.sh [OUTPUT_DIRECTORY [HTML_OUTPUT]]
#
# With no arguments the HTML goes to docs/html (the Doxyfile default) — used by
# the local build, the CMake `docs` target, and the CI check. The website
# deploy passes an output directory and HTML subdir to publish under /docs/api.
# PROJECT_NUMBER is stamped from the project() VERSION in CMakeLists.txt here so
# the version lives in exactly one place rather than being re-parsed by every
# consumer.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

version=$(awk '$1 == "VERSION" { print $2; exit }' CMakeLists.txt)

if [ -z "$version" ]; then
  echo "gen-docs.sh: no project() VERSION in CMakeLists.txt" >&2
  exit 1
fi

if [ $# -ge 1 ]; then mkdir -p "$1"; fi

{
  cat .config/Doxyfile
  echo "PROJECT_NUMBER = $version"
  if [ $# -ge 1 ]; then echo "OUTPUT_DIRECTORY = $1"; fi
  if [ $# -ge 2 ]; then echo "HTML_OUTPUT = $2"; fi
} | doxygen -
