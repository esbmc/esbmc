#!/bin/sh

BENCHEXEC_BIN=/usr/bin/benchexec
BENCHEXEC_COMMON_FLAGS="-o $HOME/witness-output/ -N 9 ./cpachecker.xml --read-only-dir / --overlay-dir /home  --container"

# Prepare Environment to run benchexec
setup_folder () {
    echo "Setting up machine folder..."
    ls $HOME/esbmc-output/
    mv $HOME/esbmc-output/*.files $HOME/witness-files
    cp esbmc-src/scripts/competitions/svcomp/cpachecker.xml $HOME/cpachecker.xml
    rm -rf $HOME/validation-action $HOME/witness-output $HOME/witness-output.zip
    mkdir $HOME/validation-action
    cd $HOME/validation-action

    curl -L https://zenodo.org/records/16682497/files/CPAchecker-4.1-unix.zip?download=1 -o cpa.zip
    unzip cpa.zip
    mv CPAchecker* cpachecker && cd cpachecker
    cp $HOME/cpachecker.xml .
    echo "Configuration done. See files below"
    ls -l
    echo
}

benchexec_run_full_set () {
    echo "$BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS"
    $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS
}

benchexec_run_set () {
    echo "$BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS -r $1"
    $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS -r $1
}

benchexec_run_task () {
    echo "$BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS -t $1"
    $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS -t $1
}

save_files () {
    cd $HOME
    zip -r witness-output.zip $HOME/witness-output
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
