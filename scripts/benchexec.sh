#!/bin/sh

BENCHEXEC_BIN=/usr/bin/benchexec
BENCHEXEC_COMMON_FLAGS="-o ../esbmc-output/ -N 30 ./esbmc.xml --read-only-dir / --overlay-dir /home  -T $TIMEOUT --container"

# Prepare Environment to run benchexec
setup_folder () {
    echo "Setting up machine folder..."
    cp esbmc-src/scripts/competitions/svcomp/esbmc-wrapper.py $HOME/esbmc-wrapper.py
    cp esbmc-src/scripts/competitions/svcomp/esbmc.xml $HOME/esbmc.xml
    rm -rf $HOME/output-action $HOME/esbmc-output esbmc-src $HOME/run-output.zip
    mkdir $HOME/output-action
    cp ./bin/esbmc $HOME/output-action
    cd $HOME/output-action
    cp $HOME/esbmc-wrapper.py .
    cp $HOME/esbmc.xml .
    chmod +x ./esbmc
    chmod +x ./esbmc-wrapper.py
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
    zip -r run-output.zip ../esbmc-output
    cp run-output.zip $HOME/output.zip
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
