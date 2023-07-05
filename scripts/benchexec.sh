#!/bin/sh

BENCHEXEC_BIN=/usr/bin/benchexec
BENCHEXEC_COMMON_FLAGS="-o ../esbmc-output/ -N $THREADS ./esbmc.xml --read-only-dir / --overlay-dir /home  -T $TIMEOUT --container"


# Prepare Environment to run benchexec
setup_folder () {
    echo "Setting up machine folder..."
    cp esbmc-src/scripts/competitions/svcomp/esbmc-wrapper.py $HOME/esbmc-wrapper.py
    rm -rf $HOME/output-action $HOME/esbmc-output esbmc-src $HOME/run-output.zip
    mkdir $HOME/output-action
    cp ./esbmc $HOME/output-action
    cd $HOME/output-action
    cp $HOME/esbmc-wrapper.py .
    cp $HOME/esbmc.xml .
    chmod +x ./esbmc
    chmod +x ./esbmc-wrapper.py
    echo "Configuration done. See files below"
    ls -l
    echo
}

benchexec_run_full_set() {
    echo "$BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS"
    $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS
}

benchexec_run_set() {
    echo "$BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS -r $set"
    $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS -r $set
}

benchexec_run_task() {
    echo "$BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS -t $task"
    $BENCHEXEC_BIN $BENCHEXEC_COMMON_FLAGS -t $task
}

save_files() {
    echo "Saving files in $out_dir"
    zip -r run-output.zip ../esbmc-output
    cp run-output.zip $HOME/$out_dir
}

# Setup build flags (release, debug, sanitizer, ...)
while getopts r:t:f flag
do
    case "${flag}" in
        r) echo "Running run-set ${OPTARG}";;
        t) echo "Running run-task ${OPTARG}";;
        f) echo "Running full-mode";;
    esac
done

