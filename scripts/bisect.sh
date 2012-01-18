#!/bin/sh

testname=$1

# Assume we're run from project root
ESBMCDIR=`pwd`

# To build: make clean, make.
make clean > /dev/null 2>/dev/null
make > /dev/null 2>/dev/null
# Make twice to ensure z3 objects propagate; due to ancient dependancy faults.
make > /dev/null 2>/dev/null

# If the build failed, curses.
if test $? != 0; then
  exit 125 # untestable
fi

# Otherwise, find the test we'll be testing.
cd regression
stat esbmc/$testname
if test $? = 0; then
  cd esbmc/$testname
else
  stat esbmc-pt/$testname
  if test $? = 0; then
    cd esbmc-pt/$testname
  else
    stat esbmc-cpp/$testname
    if test $? != 0; then
      echo "Can't find $testname" >&2
      exit 125
    fi
    cd esbmc-cpp/$testname
  fi
fi
