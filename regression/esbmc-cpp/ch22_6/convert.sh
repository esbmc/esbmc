#!/bin/bash
RESULT=`cat testllvm.desc |head -n 3 | tail -n 1`
#echo $RESULT

if [ "$RESULT" = "^CONVERSION ERROR$" ]
  then {
	 $(cd ..)
	}
   else {
   	$(clang++ -w -emit-llvm *.cpp -c -g)
	$(llvm-link *.o -o main.ll -S)
	$(llc -march=c main.ll -o main.c)
   }
fi
