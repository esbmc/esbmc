#!/bin/sh

if test ! -e esbmc; then
  echo "Please run from src/ dir";
  exit 1
fi

libtoolize

# /usr/share contains PKG macros
aclocal -I scripts/build-aux/m4 -I /usr/share/aclocal

automake --add-missing --foreign

autoconf

rm -rf autom4te.cache
