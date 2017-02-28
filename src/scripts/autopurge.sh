#!/bin/sh

if test ! -e esbmc; then
  echo "Please run from src/ dir";
  exit 1
fi

# Remove everything installed by automake / libtool etc -- they're all symlinks
# as of this moment.

topdir=`pwd`
cd scripts/build-aux

find . -type l | xargs rm 2>/dev/null

cd $topdir

# Remove other guff that autotools generates
rm -rf autom4te.cache 2>/dev/null
rm aclocal.m4 2>/dev/null
rm configure 2>/dev/null
rm config.log 2>/dev/null
rm config.status 2>/dev/null
rm stamp-h1 2>/dev/null
rm ac_config.h 2>/dev/null
rm libtool 2>/dev/null
rm scripts/ac_config.in~ 2>/dev/null
rm -rf libltdl 2>/dev/null


find . | grep Makefile.in | grep -v regression | xargs rm 2>/dev/null
