#!/bin/sh

symname=$1

thefile=`mktemp /tmp/ESBMCbuildXXXXXX`
echo "char $1 [] = {" >> $thefile
cat $2 | od -v -t d1 stddef.h | sed -E 's_^[0-9]+( +)?__' | sed -E 's_ +_ _g' | sed 's_ _,_g' | sed -E 's_([0-9])$_\1,_g' >> $thefile
echo "};" >> $thefile
mv $thefile $3
