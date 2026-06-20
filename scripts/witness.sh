#!/bin/sh

BENCHEXEC_BIN=/usr/bin/benchexec
BENCHEXEC_COMMON_FLAGS="-o $HOME/witness-output/ -N 9 --read-only-dir / --overlay-dir /home/benchexec --container"

# Prepare environment for the selected validator (CPAChecker or ESBMC).
# Witness files are moved from esbmc-output once here.
setup () {
    echo "Setting up $VALIDATOR validation..."
    rm -rf $HOME/witness-files/*
    mv $HOME/esbmc-output/*.files/* $HOME/witness-files
    rm -rf $HOME/witness-output $HOME/witness-output.zip

    if [ "$VALIDATOR" = "CPAChecker" ]; then
        cp esbmc-src/scripts/competitions/svcomp/witness/cpachecker-violation-v2.xml $HOME/cpachecker-violation-v2.xml
        cp esbmc-src/scripts/competitions/svcomp/witness/cpachecker-correctness-v2.xml $HOME/cpachecker-correctness-v2.xml
        rm -rf $HOME/validation-action
        mkdir $HOME/validation-action
        cd $HOME/validation-action
        curl -sSL https://zenodo.org/records/17777566/files/CPAchecker-4.2.2-unix.zip?download=1 -o cpa.zip
        unzip -q cpa.zip
        mv CPAchecker* cpachecker && cd cpachecker
        cp $HOME/cpachecker-violation-v2.xml .
        cp $HOME/cpachecker-correctness-v2.xml .
    elif [ "$VALIDATOR" = "ESBMC" ]; then
        cp esbmc-src/scripts/competitions/svcomp/witness/esbmc-violation-v2.xml $HOME/esbmc-violation-v2.xml
        cp esbmc-src/scripts/competitions/svcomp/witness/esbmc-correctness-v2.xml $HOME/esbmc-correctness-v2.xml
        export PYTHONPATH="$(pwd)/esbmc-src/scripts/competitions/svcomp/witness${PYTHONPATH:+:$PYTHONPATH}"
        # ESBMC binary and esbmc-wrapper.py already in $HOME/output-action/ from benchexec.sh
        cd $HOME/output-action
        cp $HOME/esbmc-violation-v2.xml .
        cp $HOME/esbmc-correctness-v2.xml .
    fi

    echo "Configuration done. See files below"
    ls -l
    echo
}

violation_xml () {
    [ "$VALIDATOR" = "CPAChecker" ] && echo "./cpachecker-violation-v2.xml" || echo "./esbmc-violation-v2.xml"
}

correctness_xml () {
    [ "$VALIDATOR" = "CPAChecker" ] && echo "./cpachecker-correctness-v2.xml" || echo "./esbmc-correctness-v2.xml"
}

# Run benchexec with the right XMLs based on WITNESS_OPTS.
# $1 is an optional extra flag like "-r <runset>" or "-t <task>".
run_benchexec () {
    if [ "$WITNESS_OPTS" = "Full" ] || [ "$WITNESS_OPTS" = "Violation" ]; then
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS $(violation_xml) $1
    fi
    if [ "$WITNESS_OPTS" = "Full" ] || [ "$WITNESS_OPTS" = "Correctness" ]; then
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS $(correctness_xml) $1
    fi
}

save () {
    cd $HOME
    zip -r witness-output.zip witness-output
}

setup

while getopts r:t:f flag
do
    case "${flag}" in
        r) echo "Running run-set ${OPTARG}"
           run_benchexec "-r ${OPTARG}";;
        t) echo "Running run-task ${OPTARG}"
           run_benchexec "-t ${OPTARG}";;
        f) echo "Running full-mode"
           run_benchexec;;
    esac
done

save
