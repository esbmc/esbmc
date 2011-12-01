#!/bin/bash

$(clang++ -w -emit-llvm *.cpp -c)
#if [ -e main.o ]
#  then {

	$(llvm-ld *.o -o main)
	$(opt -std-compile-opts main.bc -o main.bc)
	$(llc -march=c main.bc -o main.c)
#   }
#   else {
#     echo "Arquivo não compilável."
#   }
#fi

