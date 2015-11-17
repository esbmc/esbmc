#!/bin/bash

# Path to the ESBMC binary
path_to_esbmc=esbmc

# Global command line, common to all (normal) tests.
global_cmd_line="--k-induction-parallel --k-step 100 --error-label ERROR --boolector"
# The simple memory model command line is the global, without all the
# safety checks.
memory_cmd_line="--no-unwinding-assertions --64 -DLDV_ERROR=ERROR -Dassert=notassert -D_Bool=int --quiet --context-switch 3 --state-hashing"

# The '-D' options are a series of workarounds for some problems encountered:
#  -DLDV_ERROR=ERROR  maps the error label in the 'regression' dir to 'ERROR',
#                     so that we don't have to parse the property file,
#  -Dassert=notassert maps the error function in 'seq-mthreaded' directory to
#                     be named 'notassert', as giving a body for the 'assert'
#                     function conflicts with our internal library
#  -D_Bool=int        works around the presence of some booleans inside
#                     bitfields in some linux-based benchmarks.

# Memsafety cmdline picked
do_memsafety=0

while getopts "c:mh" arg; do
    case $arg in
        h)
            echo "Usage: $0 [options] path_to_benchmark
Options:
-h             Print this message
-c propfile    Specifythe given property file
-m             Use the 'simple' memory model (i.e., the memsafety tests)"
            ;;
        c)
            # Given the lack of variation in the property file... we don't
            # actually interpret it. Instead we have the same options to all
            # tests (except for the memory checking ones), and define all the
            # error labels from other directories to be ERROR.
            ;;
        m)
            do_memsafety=1
            ;;
    esac
done

# Store the path to the file we're going to be checking.
benchmark=$1

if test "${benchmark}" = ""; then
    echo "No benchmark given" #>&2
    exit 1
fi

# Pick the command line to be using
if test ${do_memsafety} = 0; then
    cmdline=${global_cmd_line}
else
    cmdline=${memory_cmd_line}
fi

# Our approach is one of iterative deepening. Run ESBMC 3 times with a deeper
# unwind bound each time. Turn this into a single command string:
deepen_cmdline="${path_to_esbmc} ${cmdline} ${benchmark};"

# Drop all output into a temporary file,
TMPFILE=`mktemp`

# Invoke our iterative deepening command, wrapped in a timeout so that we can
# postprocess the results. `timeout` is part of coreutils on debian and fedora.
timeout 900 bash -c "$deepen_cmdline" > ${TMPFILE} 2>/dev/null

# Postprocessing: first, collect some facts
grep "VERIFICATION FAILED" ${TMPFILE} > /dev/null #2>&1
failed=$?
grep "VERIFICATION SUCCESSFUL" ${TMPFILE} > /dev/null #2>&1
success=$?
grep "Timed out" ${TMPFILE} > /dev/null #2>&1
timeout=$?

#echo "FAILED = $failed, SUCCESS = $success, TIMEOUT = $timeout"

# Decide which result we determined here. The ordering is important: check for
# a counterexample first. The output file may contain both success and failure,
# if a smaller unwind bound didn't uncover the error. But if there's a
# counterexample, then there's an error.
if test $failed = 0; then
    # Error path found
    
    IS_PARALLEL_K_INDUCTION=$(echo $global_cmd_line | grep -o "k-induction-parallel" | wc -l )    
    if [ $IS_PARALLEL_K_INDUCTION -eq 1 ]; then
	RESULT=$(tac ${TMPFILE} | head -n2 );
	TYPE=$(echo $(echo $RESULT | sed -e "s:VERIFICATION FAILED Solution found by the ::"))
        echo "FALSE - $TYPE";
    else
        TYPE=$(grep -G "\*\*\* Checking" ${TMPFILE} | tac | head -n1 | tr '[:upper:]' '[:lower:]' | sed -e "s:\*\*\* checking ::")
        echo "FALSE - $TYPE"
    fi
   
    echo "Counterexample is in ${TMPFILE}"
elif test $success = 0; then
    
    IS_PARALLEL_K_INDUCTION=$(echo $global_cmd_line | grep -o "k-induction-parallel" | wc -l )    
    if [ $IS_PARALLEL_K_INDUCTION -eq 1 ]; then
	RESULT=$(tac ${TMPFILE} | head -n2 );
	TYPE=$(echo $(echo $RESULT | sed -e "s:VERIFICATION SUCCESSFUL Solution found by the ::"))
    else
        TYPE=$(grep -G "\*\*\* Generating" ${TMPFILE} | tac | head -n1 | tr '[:upper:]' '[:lower:]' | sed -e "s:\*\*\* generating ::" | sed -e "s: \*\*\*::")
    fi

    echo "TRUE - $TYPE"
    # Clean up after ourselves
    rm ${TMPFILE}
elif test $timeout = 0; then
   echo "TIMEOUT"
else
   echo "UNKNOWN"
fi
