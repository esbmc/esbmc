#!/bin/bash

# Originally from: https://github.com/fbrausse/esbmc/blob/sv-comp-results/gh-744/svcomp/run-tg
if [ $# -ne 2 ]; then
    echo "usage: $0 OLD NEW"
    exit 1
fi

shopt -s failglob

old=$1
new=$2
out=diff-$1-$2

res=()
mkdir $out &&
    for c in no-overflow termination unreach-call valid-memcleanup.MemSafety-MemCleanup valid-memsafety; do
	table-generator -f html -d -o $out/$c {$old,$new}/esbmc-kind.*.results.SV-COMP22_$c.xml.bz2 &&
	    f=`realpath --relative-to "$(dirname "$0")" $out/$c/results.*.diff.html` &&
	    res+=( "- [$c](https://fbrausse.github.io/esbmc/svcomp/$f)" )
    done
echo
printf "%s\n" "${res[@]}"
