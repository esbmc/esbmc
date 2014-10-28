#!/bin/bash

# Path to ESBMC binary
ESBMC_PATH=../esbmc
# A temporary file to pipe output into for later examination
TMPFILE=`mktemp`

if test $# -ne 1; then
        echo "usage: esbmc_wrapper_script.sh filename" >&2
        exit 1
fi

# Two arguments: the directory name and the path to the file to check.
dirname=$1
filename=$1

# Options we might run with.
NORMALOPTS=" --unwind 6 --no-unwinding-assertions --64 --no-assertions -D_Bool=int --no-bounds-check --no-pointer-check --no-div-by-zero-check --error-label ERROR -DLDV_ERROR=ERROR -Dassert=notassert --quiet --context-switch 3 --state-hashing --no-assertions"
MEMSAFETYOPTS=" --unwind 6 --no-unwinding-assertions --64 --no-assertions -D_Bool=int --memory-leak-check -DLDV_ERROR=ERROR -Dassert=notassert --quiet --context-switch 3 --state-hashing"

# There are two modes we might run in: Normal, and memsafety. The first is just
# a general set of options that work well for all testcases, the second is
# options specific to checking the memory safety category. The drivers64
# catagory receives no special options.

# Decide what options to run with.

##if test "$dirname" = "memsafety"; then
if [ $(echo $dirname | grep memsafety | wc -l) -eq 1 ]; then
	THEOPTS=$MEMSAFETYOPTS
else
	THEOPTS=$NORMALOPTS
fi

run_bmc()
{
	$ESBMC_PATH $THEOPTS --timeout 895s $filename 2>&1 | tee $TMPFILE > /dev/null
}

# Run ESBMC in k-induction mode for half the time
$ESBMC_PATH $THEOPTS $filename --timeout 450s --k-induction-parallel --k-step 100 --memlimit 15g 2>&1 | tee $TMPFILE > /dev/null

# Did we produce a concrete answer?
grep "VERIFICATION SUCCESSFUL" $TMPFILE > /dev/null
UNREACH=$?
grep "VERIFICATION FAILED" $TMPFILE > /dev/null
REACH=$?
grep "VERIFICATION UNKNOWN" $TMPFILE > /dev/null
UNKNOWN=$?

# If we produced a concrete answer, use that. Otherwise, run in BMC mode for the
# rest of the allocated time. All pthread tests will be rejected by
# ESBMC in k-induction mode.
if test $UNKNOWN -eq 0; then
	# No answer could be produced; run in BMC mode.
	run_bmc
elif test $UNREACH -eq 0; then
#	# We've decided that the property is unreachable. Report and exit.
	echo ""
#	exit 0
elif test $REACH -eq 0; then
#	# We've decided that the property is reachable. Report and exit.
	echo ""
#	exit 0
else
	# Something bad happened; default to running BMC mode.
	run_bmc
fi

# Postprocessing: first, collect some facts
grep "VERIFICATION FAILED" ${TMPFILE} > /dev/null #2>&1
failed=$?
grep "VERIFICATION SUCCESSFUL" ${TMPFILE} > /dev/null #2>&1
success=$?
grep "Timed out" ${TMPFILE} > /dev/null #2>&1
timeout=$?

#echo " FAILED=$failed SUCCESS=$success TIMEOUT=$timeout";

# Decide which result we determined here. The ordering is important: check for
# a counterexample first. The output file may contain both success and failure,
# if a smaller unwind bound didn't uncover the error. But if there's a
# counterexample, then there's an error.
if test $timeout = 0; then
   echo "TIMEOUT"
elif test $failed = 0; then
    # Error path found    
    IS_PARALLEL_K_INDUCTION=$(echo $global_cmd_line | grep -o "k-induction-parallel" | wc -l )    
    if [ $IS_PARALLEL_K_INDUCTION -eq 1 ]; then
	RESULT=$(tac ${TMPFILE} | head -n2 );
	TYPE=$(echo $(echo $RESULT | sed -e "s:VERIFICATION FAILED Solution found by the ::"))
    else
        TYPE=$(grep -G "\*\*\* Checking" ${TMPFILE} | tac | head -n1 | tr '[:upper:]' '[:lower:]' | sed -e "s:\*\*\* checking ::")     
    fi
    if [ -z "$TYPE"  ]; then
       TYPE="BMC"
    fi    
    echo "FALSE - $TYPE"    
    echo "Counterexample is in ${TMPFILE}"
elif test $success = 0; then    
    IS_PARALLEL_K_INDUCTION=$(echo $global_cmd_line | grep -o "k-induction-parallel" | wc -l )    
    if [ $IS_PARALLEL_K_INDUCTION -eq 1 ]; then
	RESULT=$(tac ${TMPFILE} | head -n2 );
	TYPE=$(echo $(echo $RESULT | sed -e "s:VERIFICATION SUCCESSFUL Solution found by the ::"))
    else
        TYPE=$(grep -G "\*\*\* Generating" ${TMPFILE} | tac | head -n1 | tr '[:upper:]' '[:lower:]' | sed -e "s:\*\*\* generating ::" | sed -e "s: \*\*\*::")
    fi
    if [ -z "$TYPE"  ]; then
        TYPE="BMC"
    fi
    echo "TRUE - $TYPE"
    # Clean up after ourselves
    rm ${TMPFILE}
else
    echo "UNKNOWN"
fi

