#!/bin/bash

if test ! -e .git; then
  echo "Please run from ESBMC root dir";
  exit 1
fi

scripts/autoboot.sh
ln -s scripts/debian debian
