#!/bin/bash
RESULT=`cat testllvm.desc |head -n 3 | tail -n 1`
#echo $RESULT

if [ "$RESULT" = "^CONVERSION ERROR$" ]
  then {
	 $(cd ..)
	}
   else {
   	$(clang++ -w -emit-llvm *.cpp -c -g -I/usr/share/qt4/mkspecs/linux-g++ -I. -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui -I/usr/include/qt4 -I. -I. -m64 -pipe -O2 -Wall -W -D_REENTRANT -DQT_WEBKIT -DQT_NO_DEBUG -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED -I/usr/share/qt4/mkspecs/linux-g++-64 -I. -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui -I/usr/include/qt4 -I. -I.)
	$(llvm-link *.o -o main.bc)
	$(llc -march=c main.bc -o main.c)
   }
fi
