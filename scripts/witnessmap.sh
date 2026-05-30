#!/bin/sh
# shellcheck disable=SC2086

BENCHEXEC_BIN=/usr/bin/benchexec
BENCHEXEC_COMMON_FLAGS="-o $HOME/esbmc-output/ --read-only-dir / --overlay-dir /home/benchexec --container"

setup () {
    echo "Setting up WitnessMap..."
    rm -rf "$HOME/witnessmap-action"
    mkdir "$HOME/witnessmap-action"
    cp esbmc-src/scripts/competitions/svcomp/witness/witnessmap-v2.xml "$HOME/witnessmap-v2.xml"
    cd "$HOME/witnessmap-action"
    curl -sSL https://gitlab.com/sosy-lab/software/witnessmap/-/raw/main/witness_map.py -o witness_map.py
    chmod +x witness_map.py
    cp "$HOME/witnessmap-v2.xml" .
    echo "WitnessMap setup done. See files below:"
    ls -l
    echo
}

setup
cd "$HOME/witnessmap-action"
echo "Running WitnessMap (full)"
"$BENCHEXEC_BIN" $BENCHEXEC_COMMON_FLAGS witnessmap-v2.xml
rm -rf "$HOME/witnessmap-action"
