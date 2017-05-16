#!/bin/sh

if test ! -e esbmc; then
  echo "Please run from src/ dir";
  exit 1
fi

autoreconf -fi

rm -rf autom4te.cache
