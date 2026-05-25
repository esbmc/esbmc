#!/bin/bash

# Originally from: https://github.com/fbrausse/esbmc/blob/sv-comp-results/gh-744/svcomp/run-tg
if [ $# -ne 2 ]; then
    echo "usage: $0 OLD NEW"
    exit 1
fi

echo "Generating diff between runs"

shopt -s nullglob

old=$1
new=$2
out=diff-$1-$2

mkdir -p "$out"

# Collect unique result identifiers (runset.task) from both directories,
# regardless of tool name or date prefix.
declare -A seen
for f in "$old"/*.results.*.xml.bz2 "$new"/*.results.*.xml.bz2; do
    base=$(basename "$f")
    id="${base#*.results.}"   # strip <tool>.<date>.results.
    id="${id%.xml.bz2}"       # strip .xml.bz2
    seen["$id"]=1
done

if [ ${#seen[@]} -eq 0 ]; then
    echo "No result files found in $old or $new"
    exit 1
fi

for id in "${!seen[@]}"; do
    files=( "$old"/*.results."$id".xml.bz2 "$new"/*.results."$id".xml.bz2 )
    [ ${#files[@]} -gt 0 ] && table-generator -f html -d -o "$out/$id" "${files[@]}"
done
echo
