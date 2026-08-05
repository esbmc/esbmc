#!/bin/bash
# Fetch and configure a Linux tree for regression/cxl-linux/.
#
# Those harnesses compile real kernel source, so they need a tree whose
# generated headers exist -- autoconf.h, asm-offsets.h and the
# arch/x86/include/generated wrappers. This script produces exactly that, and
# nothing more: it never builds the kernel.
#
# Usage: cxl_prepare_kernel.sh [srcdir] [builddir]
#   CXL_KERNEL_VERSION   kernel to fetch (default below)
#
# The version is pinned rather than tracking mainline. The harnesses assert
# properties of specific functions at specific lines; a moving target would
# turn an unrelated refactor upstream into a CI failure here, which is noise
# rather than signal. Bump it deliberately, and re-check the verdict table in
# regression/cxl-linux/README.md when you do.
set -euo pipefail

VERSION=${CXL_KERNEL_VERSION:-7.1.5}
SRC=${1:-$PWD/linux-$VERSION}
BUILD=${2:-$PWD/linux-build-cxl}
MAJOR=${VERSION%%.*}

if [ ! -d "$SRC" ]; then
  TARBALL="linux-$VERSION.tar.xz"
  URL="https://cdn.kernel.org/pub/linux/kernel/v${MAJOR}.x/$TARBALL"
  echo "==> fetching $URL"
  mkdir -p "$(dirname "$SRC")"
  curl -fsSL "$URL" -o "/tmp/$TARBALL"
  echo "==> unpacking into $SRC"
  mkdir -p "$SRC"
  tar -xf "/tmp/$TARBALL" -C "$SRC" --strip-components=1
  rm -f "/tmp/$TARBALL"
else
  echo "==> reusing existing source tree $SRC"
fi

mkdir -p "$BUILD"

echo "==> configuring (out of tree, so the source stays pristine)"
make -C "$SRC" O="$BUILD" ARCH=x86_64 defconfig >/dev/null

"$SRC"/scripts/config --file "$BUILD/.config" \
    -e CXL_BUS -e CXL_PCI -e CXL_MEM -e CXL_PORT -e CXL_ACPI

make -C "$SRC" O="$BUILD" ARCH=x86_64 olddefconfig >/dev/null

echo "==> make prepare"
# objtool needs libelf headers and is not installed everywhere. Its failure is
# harmless here: every generated header ESBMC needs is produced before objtool
# is built. Checking for the headers afterwards is what actually decides
# whether this worked, so a non-zero exit is not by itself fatal.
make -C "$SRC" O="$BUILD" ARCH=x86_64 prepare >/dev/null 2>&1 || \
  echo "    (make prepare returned non-zero; verifying headers instead)"

# CONFIG_CC_HAS_COUNTED_BY is a compiler-capability probe, so its value depends
# on whichever compiler ran defconfig rather than on anything we pin. When it is
# set, struct members carry __attribute__((__counted_by__(m))), and ESBMC cannot
# convert the resulting CountAttributedType -- every harness dies with
# "Conversion of unsupported clang type: CountAttributed" before verifying
# anything. Clearing the define makes compiler_types.h take its #else branch and
# expand __counted_by() to nothing.
#
# This drops a bounds *annotation*, not a bounds *check*: the attribute only
# feeds CONFIG_UBSAN_BOUNDS and CONFIG_FORTIFY_SOURCE, neither of which this
# config enables. Record it as an assumption all the same -- see the stub
# surface section of regression/cxl-linux/README.md.
AUTOCONF="$BUILD/include/generated/autoconf.h"
if [ -f "$AUTOCONF" ]; then
  sed -i '/^#define CONFIG_CC_HAS_COUNTED_BY\(_PTR\)\? 1$/d' "$AUTOCONF"
fi

missing=0
for h in "$BUILD/include/generated/autoconf.h" \
         "$BUILD/include/generated/asm-offsets.h" \
         "$BUILD/arch/x86/include/generated/asm" ; do
  if [ ! -e "$h" ]; then
    echo "MISSING: $h" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "cxl_prepare_kernel.sh: generated headers absent; the harnesses cannot compile" >&2
  exit 1
fi

echo "==> ready"
echo "    CXL_LINUX_SRC=$SRC"
echo "    CXL_LINUX_BUILD=$BUILD"
