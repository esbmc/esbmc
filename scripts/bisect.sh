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
stat esbmc/$testname >/dev/null 2>&1
if test $? = 0; then
  cd esbmc/$testname
else
  stat esbmc-pt/$testname > /dev/null 2>&1
  if test $? = 0; then
    cd esbmc-pt/$testname
  else
    stat esbmc-cpp/$testname > /dev/null 2>&1
    if test $? != 0; then
      echo "Can't find $testname" >&2
      exit 125
    fi
    cd esbmc-cpp/$testname
  fi
fi

counter=0
args=""
declare -a regexarr
data=`cat test.desc`
while read line
do
  if test $counter -lt 2; then
    args="$args $line"
  else
    regexarr[$(($counter-2))]="$line"
  fi
  counter=$(($counter+1))
done < test.desc

# Actually run esbmc
tmpfile=`mktemp`
$ESBMCDIR/esbmc/esbmc $args > $tmpfile 2>&1
