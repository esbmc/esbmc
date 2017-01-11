#!/bin/sh

if test ! -e esbmc; then
  echo "Please run from src/ dir";
  exit 1
fi

libtoolize

aclocal -I scripts/build-aux/m4

automake --add-missing --foreign

autoconf

rm -rf autom4te.cache
