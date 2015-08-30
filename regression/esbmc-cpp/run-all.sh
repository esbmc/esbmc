#!/bin/bash

echo ""
echo "Script LLBMC Started:" $(date +"%T")
echo ""

MODULES="algorithm cpp deque inheritance list queue stack stream string try_catch vector"

for module in $MODULES; do
  echo "============================== "
  echo -n "Running" $module...
  START=$(date +"%s")
  sh llbmc.sh $module > logs/llbmc.$module.log
  echo "Done!"
  END=$(date +"%s")

  echo "Time elapsed: " $(( $END - $START )) "s"
  echo "Time elapsed: " $(( $END - $START )) "s" >> logs/llbmc.$module.log
done
echo "============================== "

echo ""
echo "Script LLBMC Ended:" $(date +"%T")
echo ""

echo ""
echo "Script ESBMC Started:" $(date +"%T")
echo ""

for module in $MODULES; do
  echo "============================== "
  echo -n "Running" $module...
  START=$(date +"%s")
  make $module > logs/esbmc.$module.log
  echo "Done!"
  END=$(date +"%s")
  echo "Time elapsed: " $(( $END - $START )) "s"
  echo "Time elapsed: " $(( $END - $START )) "s" >> logs/esbmc.$module.log
done
echo "============================== "

echo ""
echo "Script ESBMC Ended:" $(date +"%T")
echo ""


