#!/bin/bash
RESULT=`cat testllvm.desc |head -n 3 | tail -n 1`
#echo $RESULT

if [ "$RESULT" = "^CONVERSION ERROR$" ]
  then {
	 $(cd ..)
	}
   else {
	$(clang++ -c -g -emit-llvm *.cpp -fno-exceptions)
	$(llvm-link *.o -o main.bc)
   }
fi
