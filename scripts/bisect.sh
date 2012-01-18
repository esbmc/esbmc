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

# Search the array of regexes we have for what we're looking for, success or
# failure.
success=0;
failure=0;
winregex='^VERIFICATION SUCCESSFUL$'
failregex='^VERIFICATION FAILED$'
for regexp in "${regexarr[@]}"
do
  echo "Looking at regex $regexp"
  if test "$regexp" = "$winregex"; then
    success=1;
    rightregex=$winregex
    wrongregex=$failregex
  fi
  if test "$regexp" = "$failregex"; then
    failure=1;
    rightregex=$failregex
    wrongregex=$winregex
  fi
done

if test $success = 1 -a $failure = 1; then
  echo "Test desc file matches both success and failure" >&2
  exit 1
fi

if test $success = 0 -a $failure = 0; then
  echo "Test desc file matches neither success nor failure in verification" >&2
  exit 1
fi

# Actually run esbmc
tmpfile=`mktemp`
$ESBMCDIR/esbmc/esbmc $args > $tmpfile 2>&1
