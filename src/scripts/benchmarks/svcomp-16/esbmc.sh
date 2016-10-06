#!/bin/bash

START=$(date +%s);

PROPERTY_FORGOTTEN_MEMORY_TAG="dereference failure: forgotten memory"
PROPERTY_INVALID_POINTER_TAG="dereference failure: invalid pointer"
PROPERTY_ARRAY_BOUND_VIOLATED_TAG="array bounds violated"
PROPERTY_ARRAY_BOUND_VIOLATED_TAG2="dereference failure: Access to object out of bounds"
PROPERTY_UNWIND_ASSERTION_LOOP_TAG="unwinding assertion loop"
PROPERTY_INVALID_FREE_TAG="Operand of free must have zero pointer offset"

cmdline=" --timeout 895s --memlimit 15g -DLDV_ERROR=ERROR -D_Bool=int --no-div-by-zero-check --boolector --force-malloc-success --unroll-loops --unwind 128 --no-unwinding-assertions "

BENCHMARK_FALSE_VALID_MEMTRACK=${PROPERTY_FORGOTTEN_MEMORY_TAG}
BENCHMARK_FALSE_VALID_FREE=${PROPERTY_INVALID_FREE_TAG}
BENCHMARK_FALSE_VALID_DEREF="(${PROPERTY_INVALID_POINTER_TAG}|${PROPERTY_ARRAY_BOUND_VIOLATED_TAG}|${PROPERTY_ARRAY_BOUND_VIOLATED_TAG2})"

# Path to the ESBMC binary
path_to_esbmc=./esbmc

# cpachecker command
cpa_command=" ./scripts/cpa.sh -disable-java-assertions -witness-check -heap 10000m -setprop cpa.arg.errorPath.graphml=error-witness.graphml "

# Benchmark result controlling flags
IS_OVERFLOW_BENCHMARK=0
IS_MEMSAFETY_BENCHMARK=0

while getopts "c:h:v" arg; do
    case $arg in
        v)
            ${path_to_esbmc} "--version"
            exit 0
            ;;
        h)
            echo "Usage: $0 [options] path_to_benchmark
Options:
-h             Print this message
-c propfile    Specifythe given property file"
            ;;
        c)
            # Given the lack of variation in the property file... we don't
            # actually interpret it. Instead we have the same options to all
            # tests (except for the memory checking ones), and define all the
            # error labels from other directories to be ERROR.
            #if ! grep -q __VERIFIER_error $OPTARG; then
            property_list=$OPTARG
            
            if grep -q -E "LTL[(]G (valid-free|valid-deref|valid-memtrack)" $OPTARG; then
              IS_MEMSAFETY_BENCHMARK=1
            elif grep -q -E "LTL[(]F end" $OPTARG; then
              echo "Unsupported Property"
              exit 1;
            elif grep -q -E "LTL[(]G ! overflow" $OPTARG; then
              IS_OVERFLOW_BENCHMARK=1
            fi

            ;;
    esac
done
shift $(( OPTIND - 1 ));

# Store the path to the file we're going to be checking.
benchmark=$1
echo "benchmark : $benchmark"

if ! [ -f "${benchmark}" ]; then
    echo "No benchmark given" #>&2
    exit 1
fi

# Add graphml informations
TMPGRAPHML="error-witness.graphml"
cmdline="$cmdline --witness-output $TMPGRAPHML"

if test $IS_OVERFLOW_BENCHMARK = 1; then
   cmdline="$cmdline --overflow-check "
fi

if test $IS_MEMSAFETY_BENCHMARK = 1; then
   cmdline="$cmdline --memory-leak-check "
else
   cmdline="$cmdline --no-pointer-check --no-bounds-check --error-label ERROR"
fi

# Drop all output into a temporary file,
TMPFILE=`mktemp`
TMPEXECLOG=`mktemp`
TMPCPALOG=`mktemp`

echo "Verifying with ESBMC "
echo "Command: ""${path_to_esbmc} ${cmdline} ${benchmark} > ${TMPFILE} 2>${TMPEXECLOG}"

${path_to_esbmc} ${cmdline} ${benchmark} > ${TMPFILE} 2>${TMPEXECLOG}

failed=0
success=0
timeout=0

END=$(date +%s);

# Postprocessing: first, collect some facts
if grep -i -q "Timed out" ${TMPFILE}; then
  timeout=1
elif grep -q "VERIFICATION FAILED" ${TMPFILE}; then

  valid_fail=0

  # memory safety validation
  if test ${IS_MEMSAFETY_BENCHMARK} = 1; then
    if grep -q -E "(${BENCHMARK_FALSE_VALID_MEMTRACK}|${BENCHMARK_FALSE_VALID_DEREF}|${BENCHMARK_FALSE_VALID_FREE})" ${TMPFILE}; then
      valid_fail=1;
    fi
  else 

    if test ${IS_OVERFLOW_BENCHMARK} != 1; then
      # Do we still have time to check the witness?
      diff_timeout=$((800 - (END-START)))

      # if there is enough time, we try to validate the witness
      if test ${diff_timeout} -gt 0; then

        # We run cpachecker in background and check its output every ten seconds because on our tests, the tool doesn't respect
        # the timelimit. Usually it hangs trying to kill the jvm for more time than specified resulting in a timeout

        cd cpachecker/
        cpa_graph=" -spec ../${TMPGRAPHML}"
        cpa_prop=" -spec ../${property_list}"
        cpa_timeout=" -timelimit ${diff_timeout} "
        echo "Checking witness with CPAChecker "
        echo "Command: ""${cpa_command} ${cpa_graph} ${cpa_prop} ${cpa_timeout} ../${benchmark} > ${TMPCPALOG} 2>${TMPEXECLOG}"
        ${cpa_command} ${cpa_graph} ${cpa_prop} ${cpa_timeout} ../${benchmark} > ${TMPCPALOG} 2>${TMPEXECLOG} &
        cd ../

        timer=0
        while [ ${timer} -lt ${diff_timeout} ]
        do
          if grep -q -E "Verification result: TRUE" ${TMPCPALOG}; then
            echo "UNKNOWN"
            exit 1;
          elif grep -q -E "Verification result: FALSE" ${TMPCPALOG}; then
            break
          elif grep -q -E "Verification result: UNKNOWN" ${TMPCPALOG}; then
            break
          elif grep -q -E "Exception" ${TMPEXECLOG}; then
            break
          fi

          sleep 10
          timer=$((timer + 10))
        done
 
      fi
    fi
  
    valid_fail=1;
  fi

  if test ${valid_fail} = 1; then
    failed=1
  fi
elif grep -q "VERIFICATION SUCCESSFUL" ${TMPFILE}; then
  success=1
fi

get_memsafety_violated_property() {
  if grep -q -E "${BENCHMARK_FALSE_VALID_MEMTRACK}" $1; then
    echo "_MEMTRACK"
  elif grep -q -E "${BENCHMARK_FALSE_VALID_DEREF}" $1; then
    echo "_DEREF"
  elif grep -q -E "${BENCHMARK_FALSE_VALID_FREE}" $1; then
    echo "_FREE"
  fi
}

# Decide which result we determined here. The ordering is important: check for
# a counterexample first. The output file may contain both success and failure,
# if a smaller unwind bound didn't uncover the error. But if there's a
# counterexample, then there's an error.
if test $timeout = 1; then
    echo "Timed Out"
elif test $failed = 1; then

    VPROP=""
    if test ${IS_OVERFLOW_BENCHMARK} = 1; then
      VPROP="_OVERFLOW"
    elif test ${IS_MEMSAFETY_BENCHMARK} = 1; then
      VPROP=$( get_memsafety_violated_property ${TMPFILE} )
    fi

    # Error path found
    echo "FALSE${VPROP}"
    echo "Counterexample in graphML format is available in: ${TMPGRAPHML}"
elif test $success = 1; then
    echo "TRUE"
    # Clean up after ourselves
else
    echo "UNKNOWN"
fi

