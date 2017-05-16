#!/bin/bash

symname=$1
symname=`basename $symname`
symname=`echo -n $symname | sed s/.hs$//`
symname="${symname}_buf"

echo "char $symname [] = {"
cat $2 | od -v -t u1 $2 | sed -E 's_^[0-9]+( +)?__' | sed -E 's_ +_ _g' | sed 's_ _,_g' | sed -E 's_([0-9])$_\1,_g'
echo "};"
echo "unsigned int ${symname}_size = sizeof($symname);"
