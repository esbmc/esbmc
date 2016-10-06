#!/bin/bash

# Path to the ESBMC binary
path_to_esbmc=./esbmc

# Global command line, common to all (normal) tests.
global_cmd_line="--64 -DLDV_ERROR=ERROR -Dassert=notassert -D_Bool=int --no-bounds-check --no-pointer-check --error-label ERROR --no-div-by-zero-check --no-assertions --quiet --context-bound 4 --force-malloc-success --boolector"

# The simple memory model command line is the global, without all the
# safety checks.
memory_cmd_line="--64 -DLDV_ERROR=ERROR -Dassert=notassert -D_Bool=int --quiet --context-bound 3 --force-malloc-success --memory-leak-check --boolector"

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
do_term=0

while getopts "c:h" arg; do
    case $arg in
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
            if ! grep -q __VERIFIER_error $OPTARG; then
                do_memsafety=1
            fi
            if ! grep -q 'LTL\\(F' $OPTARG; then
                do_term=1
            fi
            ;;
    esac
done

shift $(( OPTIND - 1 ));

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

# If we're checking for termination, then just leave unwinding assertions on.
# Otherwise, turn them off.
if test ${do_term} = 0; then
    cmdline="${cmdline} --no-unwinding-assertions"
fi

# Add graphml informations
TMPGRAPHML="./witness.graphml"
cmdline="$cmdline --witness-output $TMPGRAPHML"

# Drop all output into a temporary file,
TMPFILE=`mktemp`

# This year we're not iteratively deepening, we're running ESBMC with a fixed
# unwind bound of 16.
${path_to_esbmc} ${cmdline} --unwind 16 ${benchmark} > ${TMPFILE} 2>/dev/null

# Postprocessing: first, collect some facts
grep -q "VERIFICATION FAILED" ${TMPFILE}
failed=$?
grep -q "VERIFICATION SUCCESSFUL" ${TMPFILE}
success=$?
grep -i -q "Timed out" ${TMPFILE}
timeout=$?

# Decide which result we determined here. The ordering is important: check for
# a counterexample first. The output file may contain both success and failure,
# if a smaller unwind bound didn't uncover the error. But if there's a
# counterexample, then there's an error.
if test $failed = 0; then
    # Error path found
    echo "FALSE"
    echo "Counterexample in graphML format is available in: ${TMPGRAPHML}"
    rm ${TMPFILE}
elif test $success = 0; then
    echo "TRUE"
    # Clean up after ourselves
    rm ${TMPFILE}
elif test $timeout = 0; then
    echo "Timed Out"
    rm ${TMPFILE}
else
    echo "UNKNOWN"
fi
