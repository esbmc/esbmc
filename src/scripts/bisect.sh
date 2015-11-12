#!/bin/sh

function usage {
  echo "Usage: bisect.sh [opts]" >&2
  echo "Options:" >&2
  echo "    -t testname    Name of regression test to run" >&2
  echo "    -p dirname     Path to regression test to run" >&2
  echo "    -s             Unstash/Stash when performing a run" >&2
  echo "    -n             No decisive output means bad revision" >&2
  echo "    -T num_secs    Test taking more than num_secs means failure" >&2
  echo "You must specify either -t or -p."
}

searchpath=-1
beatstash=0
nooutputisfail=0
dirpath=""
numsecs=0
while getopts "t:p:sT:" opt; do
  case $opt in
    t)
      testname=$OPTARG
      searchpath=1
      ;;
    p)
      searchpath=0
      dirpath=$OPTARG
      ;;
    s)
      beatstash=1
      ;;
    n)
      nooutputisfail=1
      ;;
    T)
      numsecs=$OPTARG
      ;;
    \?)
      echo "Invalid option -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires argument" >&2
      exit 1
      ;;
  esac
done

if test $searchpath = -1; then
  echo "You must specify -t or -p" >&2
  exit -1
fi

# Assume we're run from project root
ESBMCDIR=`pwd`

if test $beatstash = 1; then
  git stash pop > /dev/null 2>&1
  if test $? != 0; then
    echo "Git stash pop failed" >&2
    exit -1
  fi
fi

function restash () {
  if test $beatstash = 1; then
    git stash > /dev/null 2>&1
    if test $? != 0; then
      echo "Git stash failed" >&2
      exit -1
    fi
  fi
}

# To build: make clean, make.
make clean > /dev/null 2>/dev/null
make > /dev/null 2>/dev/null
# Make twice to ensure z3 objects propagate; due to ancient dependancy faults.
make > /dev/null 2>/dev/null

# If the build failed, curses.
if test $? != 0; then
  echo "Failed to build this rev" >&2
  restash
  make clean > /dev/null 2>/dev/null
  exit 125 # untestable
fi

# Otherwise, find the test we'll be testing.
if test $searchpath = 1; then
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
        restash
        exit -1
      fi
      cd esbmc-cpp/$testname
    fi
  fi
else
  cd $dirpath
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
  restash
  exit -1
fi

if test $success = 0 -a $failure = 0; then
  echo "Test desc file matches neither success nor failure in verification" >&2
  restash
  exit -1
fi

# Potentially run with a timeout
alarm() { perl -e 'sleep shift; exec @ARGV' "$@" &  }

alarmed_exit() {
  echo "Timed out" >&2
  kill -9 $esbmcpid
  exit 1
}

trap alarmed_exit SIGINT

if test $numsecs != 0; then
  alarm $numsecs kill -s SIGINT $$
  alarmprocpid=$!
fi

# Actually run esbmc
tmpfile=`mktemp`
$ESBMCDIR/esbmc/esbmc $args > $tmpfile 2>&1 &
esbmcpid=$!

wait $esbmcpid

if test $numsecs != 0; then
  kill -9 $alarmprocpid
fi

# Sadly due to binaries in the tree, git bisect can complain if we don't clean
# immediately.
make clean > /dev/null 2>&1

restash

# Look for success
grep "$rightregex" $tmpfile
if test $? = 0; then
  rm $tmpfilet
  echo "Correct output" >&2
  exit 0 # success
fi

# Was opposite true?
grep "$wrongregex" $tmpfile
if test $? = 0; then
  rm $tmpfile
  echo "Wrong output" >&2
  exit 1 # Failure; wrong outcome
fi

# Otherwise, something crashed or went wrong.
rm $tmpfile
if test $nooutputisfail = 0; then
  echo "Noncommital output" >&2
  exit 125
else
  echo "Crash or abnormal exit" >&2
  exit 1
fi
