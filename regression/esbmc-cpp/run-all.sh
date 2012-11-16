#!/bin/bash

echo ""
echo "Script LLBMC Started:" $(date +"%T")
echo ""

echo -n "Algorithm... "
START=$(date +"%s")
sh llbmc.sh esbmc-algorithm/ > log-algorithm
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "CPP... "
START=$(date +"%s")
sh llbmc.sh esbmc-cpp/ > log-cpp
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "Deque... "
START=$(date +"%s")
sh llbmc.sh esbmc-deque/ > log-deque
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "Inheritance... "
START=$(date +"%s")
sh llbmc.sh esbmc-inheritance/ > log-inheritance
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "List... "
START=$(date +"%s")
sh llbmc.sh esbmc-list/ > log-list
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "Map... "
START=$(date +"%s")
sh llbmc.sh esbmc-map/ > log-map
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "Multiset... "
START=$(date +"%s")
sh llbmc.sh esbmc-multiset/ > log-multiset
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "Priority Queue... "
START=$(date +"%s")
sh llbmc.sh esbmc-priority_queue/ > log-priority-queue
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "Queue... "
START=$(date +"%s")
sh llbmc.sh esbmc-queue/ > log-queue
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "Set... "
START=$(date +"%s")
sh llbmc.sh esbmc-set/ > log-set
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "Stack... "
START=$(date +"%s")
sh llbmc.sh esbmc-stack/ > log-stack
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "Stream... "
START=$(date +"%s")
sh llbmc.sh esbmc-stream/ > log-stream 2>>log-stream
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "String... "
START=$(date +"%s")
sh llbmc.sh esbmc-string/ > log-string
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "Try catch... "
START=$(date +"%s")
sh llbmc.sh esbmc-try_catch/ > log-trycatch
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo -n "Vector... "
START=$(date +"%s")
sh llbmc.sh esbmc-vector/ > log-vector
echo "done!"
END=$(date +"%s")
echo "Time elapsed: " $(( $END - $START )) "s"
echo ""

echo ""
echo "Script LLBMC Ended:" $(date +"%T")
echo ""

echo ""
echo "Script ESBMC Started:" $(date +"%T")
echo ""

make all

echo ""
echo "Script ESBMC Ended:" $(date +"%T")
echo ""


