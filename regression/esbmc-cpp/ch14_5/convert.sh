#!/bin/bash
RESULT=`cat testllvm.desc |head -n 3 | tail -n 1`
#echo $RESULT

if [ "$RESULT" = "^CONVERSION ERROR$" ]
  then {
	 $(cd ..)
	}
   else {
   	$(clang++ -w -emit-llvm *.cpp -c)
	$(llvm-ld *.o -o main)
	$(opt -std-compile-opts main.bc -o main.bc)
	$(llc -march=c main.bc -o main.c)
   }
fi
