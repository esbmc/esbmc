#!/bin/sh

if test ! -e .git; then
  echo "Please run from ESBMC root dir";
  exit 1
fi

aclocal -I scripts/build-aux/m4

libtoolize

automake --add-missing --foreign

autoconf
