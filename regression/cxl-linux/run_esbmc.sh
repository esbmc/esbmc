#!/bin/bash
# Usage: run_esbmc.sh <harness.c> [extra esbmc flags...]
#
# Paths come from the environment so the same script serves a developer's
# checkout and CI. The defaults are one developer's layout and are expected to
# be wrong for everyone else -- override them rather than editing this file.
#
#   CXL_LINUX_SRC    kernel source tree            (scripts/cxl_prepare_kernel.sh)
#   CXL_LINUX_BUILD  its out-of-tree build dir     (same)
#   ESBMC            the esbmc binary
L=${CXL_LINUX_SRC:-/var/home/rafaelsa/Documents/linux-7.1.5}
B=${CXL_LINUX_BUILD:-/var/home/rafaelsa/Documents/linux-build-cxl}
E=${ESBMC:-/var/home/rafaelsa/Documents/esbmc-cxl/build/src/esbmc/esbmc}

for p in "$L" "$B"; do
  if [ ! -d "$p" ]; then
    echo "run_esbmc.sh: no such directory: $p" >&2
    echo "Set CXL_LINUX_SRC and CXL_LINUX_BUILD, or run" >&2
    echo "  scripts/cxl_prepare_kernel.sh <srcdir> <builddir>" >&2
    exit 2
  fi
done
if ! command -v "$E" >/dev/null 2>&1 && [ ! -x "$E" ]; then
  echo "run_esbmc.sh: esbmc not executable: $E (set ESBMC)" >&2
  exit 2
fi

# Resolve before the cd: the include paths below are relative to the kernel
# tree, so we have to run from there, which would otherwise strip the meaning
# out of a harness path relative to the caller's directory.
HARNESS="$(realpath "$1")"; shift
cd "$L" || exit 1
exec "$E" "$HARNESS" \
  --fms-extensions \
  -D__KERNEL__ -DKBUILD_MODNAME='"cxl_core"' -DKBUILD_BASENAME='"pci"' \
  -I . \
  -I arch/x86/include -I "$B/arch/x86/include/generated" \
  -I include -I "$B/include" \
  -I arch/x86/include/uapi -I "$B/arch/x86/include/generated/uapi" \
  -I include/uapi -I "$B/include/generated/uapi" \
  -I drivers/cxl -I drivers/cxl/core \
  "$@"
