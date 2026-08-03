#!/bin/bash
# Usage: run_esbmc.sh <harness.c> [extra esbmc flags...]
L=/var/home/rafaelsa/Documents/linux-7.1.5
B=/var/home/rafaelsa/Documents/linux-build-cxl
E=/var/home/rafaelsa/Documents/esbmc-cxl/build/src/esbmc/esbmc
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
