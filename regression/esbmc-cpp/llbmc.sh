#!/bin/bash
# Usage: esbmcwrapper path_to_source_file

SOURCES=$1/*/
TIMEOUT=3600 #15m=900s

POS_CONT=0
FAL_CONT=0
FAL_POS_CONT=0
FAL_NEG_CONT=0
CRASH_CONT=0
CANT_CONVERT_EXCEPTION_CONT=0
MEMORY_OUT_CONT=0
KILLED_CONT=0

ulimit -t $TIMEOUT
ulimit -v 22000000

TOTAL=0;

ROOT=`pwd`

for file in $SOURCES
do
  TOTAL=$((TOTAL+1))

  TMPFILE=`mktemp`
  TMPFILE1=`mktemp`
  OPTS="--ignore-missing-function-bodies --no-max-loop-iterations-checks --max-loop-iterations="
  echo
  echo "===================================================================================="
  echo "Running program" $file/main.bc
  echo "===================================================================================="
  echo

  echo "Converting... "
  cd $file/
  /usr/bin/clang++ -c -g -emit-llvm *.cpp -fno-exceptions 2> $TMPFILE

  if [ -f main.o ];
  then
    /usr/bin/llvm-link *.o -o main.bc
  fi

  cd $ROOT

  ########### EXCEPTION
  RESULT=`cat $TMPFILE | grep 'cannot use '`
  if [ "$RESULT" != "" ];
  then
    echo "CANT_CONVERT_EXCEPTION++ CLANG"
    CANT_CONVERT_EXCEPTION_CONT=$((CANT_CONVERT_EXCEPTION_CONT+1))
    continue
  fi
  ########### EXCEPTION

  if [ -f $file/main.bc ]; # the file exists
  then

    unwind=`cat $file/test.desc | grep unwind | awk -F"--unwind" '{ print $2 }' | cut -d' ' -f2`

    if [ "$unwind" = "" ]
    then
      unwind=10
    fi

    OPTS=$OPTS$unwind

    echo
    echo "/usr/bin/llbmc $OPTS $file/main.bc  $TMPFILE"
    /usr/bin/llbmc $OPTS $file/main.bc > $TMPFILE 2> $TMPFILE1

    cat $TMPFILE1

    ############# TIMEOUT
    RESULT=`cat $TMPFILE1 | grep 'Killed'`

    if [ "$RESULT" != "" ]; # LLBMC TIMEOUT :(
    then
      echo "KILLED_CONT++ TIMEOUT"
      KILLED_CONT=$((KILLED_CONT+1))
      continue
    fi
    ############# TIMEOUT

    ############# SEG FAULT
    RESULT=`cat $TMPFILE1 | grep 'fault'`

    if [ "$RESULT" != "" ]; 
    then
      echo "CRASH_CONT++ SEG FAULT"
      CRASH_CONT=$((CRASH_CONT+1))
      continue
    fi
    ############# SEG FAULT

    ############# BAD ALLOC
    RESULT=`cat $TMPFILE1 | grep 'bad_alloc'`

    if [ "$RESULT" != "" ]; # LLBMC BAD ALLOC :(
    then
      echo "MEMORY_OUT_CONT++ BAD ALLOC"
      MEMORY_OUT_CONT=$((MEMORY_OUT_CONT+1))
      continue
    fi
    ############# BAD ALLOC

    ########### SUCCESSFUL
    RESULT=`cat $TMPFILE | grep '^No error detected.$'`
    TARGET=`cat $file/testllvm.desc | grep 'SUCCESSFUL'`

    if [ "$RESULT" != "" ];
    then

      if [ "$TARGET" != "" ];
      then
        echo "POS_CONT++ SUCCESSFUL"
        POS_CONT=$((POS_CONT+1))
      else
        echo "FAL_POS_CONT++ SUCCESSFUL"
        FAL_POS_CONT=$((FAL_POS_CONT+1))
      fi

    fi

    RESULT=""
    TARGET=""
    ########### SUCCESSFUL

    ########### FAILED
    RESULT=`cat $TMPFILE | grep '^Error detected.$'`
    TARGET=`cat $file/testllvm.desc | grep 'FAILED'`

    if [ "$RESULT" != "" ];
    then

      if [ "$TARGET" != "" ];
      then
        echo "FAL_CONT++ FAILED"
        FAL_CONT=$((FAL_CONT+1))
      else
        echo "FAL_NEG_CONT++ FAILED"
        FAL_NEG_CONT=$((FAL_NEG_CONT+1))
      fi

    fi

    RESULT=""
    TARGET=""
    ########### FAILED

    ############# LLBMC CRASH
    RESULT=`cat $TMPFILE | grep 'Unrecoverable error'`

    if [ "$RESULT" != "" ]; # LLBMC crashed :(
    then
      echo "CRASH_CONT++ CRASH"
      CRASH_CONT=$((CRASH_CONT+1))
    fi
    ############# LLBMC CRASH

  else

    TARGET=`cat $file/testllvm.desc | grep 'PARSING'`
    if [ "$TARGET" != "" ];
    then

      echo "POS_CONT++ PARSING"
      POS_CONT=$((POS_CONT+1))

    else

      TARGET=`cat $file/testllvm.desc | grep 'CONVERSION'`
      if [ "$TARGET" != "" ];
      then

        echo "POS_CONT++ CONVERSION"
        POS_CONT=$((POS_CONT+1))

      else
 
        echo "CLANG BUG!" "CRASH_CONT++ CRASH"
        CRASH_CONT=$((CRASH_CONT+1))

      fi

    fi

  fi


done

echo
echo
echo "====================="
echo "TOTAL: \t\t" $TOTAL
echo
echo "POS: \t\t" $POS_CONT
echo "FAL: \t\t" $FAL_CONT
echo "FAL_POS: \t" $FAL_POS_CONT
echo "FAL_NEG: \t" $FAL_NEG_CONT
echo "CRASH: \t\t" $((CRASH_CONT+CANT_CONVERT_EXCEPTION_CONT))
echo "KILLED: \t" $KILLED_CONT
echo "MEMORY OUT: \t" $MEMORY_OUT_CONT
echo "====================="
echo
