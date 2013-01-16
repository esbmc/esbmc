#!/bin/bash

echo ""
echo "Script LLBMC Started:" $(date +"%T")
echo ""

MODULES="esbmc-algorithm esbmc-cpp esbmc-deque esbmc-inheritance esbmc-list esbmc-queue esbmc-stack esbmc-stream esbmc-string esbmc-try_catch esbmc-vector"

for module in $MODULES; do
  echo "============================== "
  echo -n "Running" $module...
  START=$(date +"%s")
  sh llbmc.sh $module > logs/llbmc.$module.log
  echo "Done!"
  END=$(date +"%s")
  echo "Time elapsed: " $(( $END - $START )) "s"
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
done
echo "============================== "

echo ""
echo "Script ESBMC Ended:" $(date +"%T")
echo ""


