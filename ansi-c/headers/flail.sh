#!/bin/sh

symname=$1

echo "char $1 [] = {"
cat $2 | od -v -t d1 stddef.h | sed -E 's_^[0-9]+( +)?__' | sed -E 's_ +_ _g' | sed 's_ _,_g' | sed -E 's_([0-9])$_\1,_g'
echo "};"
echo "unsigned int $1_size = sizeof($1);"
