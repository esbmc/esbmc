#!/bin/sh

if test ! -e .git; then
  echo "Please run from ESBMC root dir";
    exit 1
fi

# Remove everything installed by automake / libtool etc -- they're all symlinks
# as of this moment.

topdir=`pwd`
cd scripts/build-aux

find . -type l | xargs rm

cd $topdir

# Remove other guff that autotools generates
rm -rf autom4te.cache 2>/dev/null
rm aclocal.m4 2>/dev/null
rm configure 2>/dev/null

find . | grep Makefile.in | grep -v regression | xargs rm
