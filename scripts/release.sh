#!/bin/bash

if test "$#" != 3; then
  echo "Usage: release.sh refname solvers64dir solvers32dir"
  exit 1
fi

# You need a 64 bit machine to fully build a release

if test `uname -m` != "x86_64"; then
  echo "Please run release.sh on a 64 bit machine"
  exit 1
fi

satdir64=$2
satdir32=$3

# Work out whether we're going to build compat versions.

buildcompat=0

which gcc34 > /dev/null 2>&1
if test $? = 0; then
  which g++34 > /dev/null 2>&1
  if test $? = 0; then
    buildcompat=1
  fi
fi

# Find whatever the current head is
CURHEAD=`git symbolic-ref HEAD`

if test $? != 0; then
  # Not checked out a symbolic ref right now
  CURHEAD=`cat .git/HEAD`
fi

# Then, checkout whatever we've been told to release
git stash > /dev/null
git co $1 > /dev/null

if test $? != 0; then
  echo "Couldn't checkout $1"
  exit 1
fi

# Install our configuration files.
cp ./scripts/config.inc .
cp ./scripts/local.inc .

# And build build build
rm -rf .release
mkdir .release

# Use 64 bit libraries
export SATDIR=$satdir64

echo "Building 64 bit ESBMC"
make clean > /dev/null 2>&1
make > /dev/null 2>&1

if test $? != 0; then
  echo "Build failed."
  exit 1
fi

cp esbmc/esbmc .release/esbmc

if test $buildcompat = 1; then
  echo "Building compat 64 bit ESBMC"
  make clean > /dev/null 2>&1
  env CC=gcc34 CXX=g++34 make > /dev/null 2>&1

  if test $? != 0; then
    echo "Build failed."
    exit 1
  fi

  cp esbmc/esbmc .release/esbmc_compat
fi

# Try for 32 bits
echo "Building 32 bit ESBMC"
export SATDIR=$satdir32

buildfor32bits=0
make clean > /dev/null 2>&1

env EXTRACFLAGS="-m32" EXTRACXXFLAGS="-m32" LDFLAGS="-m elf_i386" make > /dev/null 2>&1

if test $? != 0; then
  echo "Buildling 32 bits failed; do you have the right headers and libraries?"
  exit 1
else
  buildfor32bits=1
  cp esbmc/esbmc .release/esbmc32
  if test $buildcompat = 1; then
    echo "Building 32 bit compat ESBMC"

    make clean > /dev/null 2>&1

    env CC=gcc34 CXX=g++34 EXTRACFLAGS="-m32" EXTRACXXFLAGS="-m32" LDFLAGS="-m elf_i386" make > /dev/null 2>&1

    if test $? != 0; then
      echo "Building 32 bit compat ESBMC failed"
      exit 1
    fi

    cp esbmc/esbmc .release/esbmc32_compat
  fi
fi

# We now have a set of binaries.
