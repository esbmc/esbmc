#!/bin/sh

# Copied from
# http://stackoverflow.com/questions/223678/which-commit-has-this-blob
# Because it wasn't worth building another script to do it

obj_name="$1"
shift
git log "$@" --pretty=format:'%T %h %s' \
| while read tree commit subject ; do
    if git ls-tree -r $tree | grep -q "$obj_name" ; then
        echo $commit "$subject"
    fi
done

