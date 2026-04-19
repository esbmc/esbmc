#!/bin/sh

BENCHEXEC_BIN=/usr/bin/benchexec
BENCHEXEC_COMMON_FLAGS="-o $HOME/witness-output/ -N 9 --read-only-dir / --overlay-dir /home/benchexec --container"
BENCHEXEC_COMMON_FLAGS_CORRECTNESS="$BENCHEXEC_COMMON_FLAGS ./cpachecker-correctness-v2.xml"
BENCHEXEC_COMMON_FLAGS_VIOLATION="$BENCHEXEC_COMMON_FLAGS ./cpachecker-violation-v2.xml"

# Prepare Environment to run benchexec
setup_folder () {
    echo "Setting up machine folder..."
    rm -rf $HOME/witness-files/*
    mv $HOME/esbmc-output/*.files/* $HOME/witness-files
    cp esbmc-src/scripts/competitions/svcomp/cpachecker-violation-v2.xml $HOME/cpachecker-violation-v2.xml
    cp esbmc-src/scripts/competitions/svcomp/cpachecker-correctness-v2.xml $HOME/cpachecker-correctness-v2.xml
    rm -rf $HOME/validation-action $HOME/witness-output $HOME/witness-output.zip
    mkdir $HOME/validation-action
    cd $HOME/validation-action

    curl -sSL https://zenodo.org/records/17777566/files/CPAchecker-4.2.2-unix.zip?download=1 -o cpa.zip
    unzip -q cpa.zip
    mv CPAchecker* cpachecker && cd cpachecker
    cp $HOME/cpachecker-violation-v2.xml .
    cp $HOME/cpachecker-correctness-v2.xml .
    echo "Configuration done. See files below"
    ls -l
    echo
}

benchexec_run_full_set () {
    echo "$BENCHEXEC_BIN"
    if [ "$WITNESS_OPTS" = "Full" ]; then
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_VIOLATION
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_CORRECTNESS
    elif [ "$WITNESS_OPTS" = "Correctness" ]; then
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_CORRECTNESS
    elif [ "$WITNESS_OPTS" = "Violation" ]; then
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_VIOLATION
    fi
}

benchexec_run_set () {
    echo "$BENCHEXEC_BIN -r $1"
    if [ "$WITNESS_OPTS" = "Full" ]; then
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_VIOLATION -r $1
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_CORRECTNESS -r $1
    elif [ "$WITNESS_OPTS" = "Correctness" ]; then
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_CORRECTNESS -r $1
    elif [ "$WITNESS_OPTS" = "Violation" ]; then
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_VIOLATION -r $1
    fi
}

benchexec_run_task () {
    echo "$BENCHEXEC_BIN -t $1"
    if [ "$WITNESS_OPTS" = "Full" ]; then
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_VIOLATION -t $1
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_CORRECTNESS -t $1
    elif [ "$WITNESS_OPTS" = "Correctness" ]; then
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_CORRECTNESS -t $1
    elif [ "$WITNESS_OPTS" = "Violation" ]; then
        $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS_VIOLATION -t $1
    fi
}

save_files () {
    cd $HOME
    zip -r witness-output.zip witness-output
}

setup_folder

# Select analysis mode
while getopts r:t:f flag
do
    case "${flag}" in
        r) echo "Running run-set ${OPTARG}"
           benchexec_run_set ${OPTARG};;
        t) echo "Running run-task ${OPTARG}"
            benchexec_run_task ${OPTARG};;
        f) echo "Running full-mode"
            benchexec_run_full_set;;
    esac
done

save_files
