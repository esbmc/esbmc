# Cross-compiling a MinGW version of ESBMC for Windows

This guide describes the procedure to obtain a Windows build of ESBMC via
cross-compilation to mingw64 on x86_64 Ubuntu-22.04.

## Prerequisites

This guide assumes that the ESBMC repository is set up in $PWD/esbmc and that
the files

- llvm+clang-14-mingw.zip
- z3-4.13.0-x64-win.zip

are available in the current directory as well. The first one should contain a
pre-built Clang-14 for mingw with the `posix` thread model.

	sudo apt-get install -y \
		mingw-w64-{i686-dev,x86-64-dev,tools} wine64 \
		gcc-mingw-w64-x86-64-posix{,-runtime} \
		g++-mingw-w64-x86-64-posix \
		gcc-mingw-w64-i686-posix-runtime \
		ninja cmake libboost1.74-tools-dev &&
	unzip llvm+clang-14-mingw.zip &&
	LLVM_OPT=-DLLVM_DIR=$PWD/llvm+clang-14-mingw/lib/cmake/llvm &&
	CLANG_OPT=-DClang_DIR=$PWD/llvm+clang-14-mingw/lib/cmake/clang &&
	unzip z3-4.13.0-x64-win.zip &&
	Z3_OPT=-DZ3_DIR=$PWD/z3-4.13.0-x64-win

## Create the toolchain file

Create a CMake toolchain file for cross-compilation
`/tmp/x86_64-w64-mingw32.toolchain.cmake` with the following contents:

	set(CMAKE_SYSTEM_NAME         Windows)
	set(CMAKE_SYSTEM_PROCESSOR    x86_64)

	set(arch                      "x86_64")
	set(os                        "w64")
	set(flavor                    "mingw32")

	set(triple                    "${arch}-${os}-${flavor}")

	set(TOOLCHAIN_PREFIX          "${triple}-")
	set(CMAKE_SYSROOT             "/usr/${triple}")

	set(CMAKE_CROSSCOMPILING_EMULATOR /tmp/${triple}.cross-emu.sh)

	set(CMAKE_ASM_COMPILER_TARGET ${triple})
	set(CMAKE_C_COMPILER_TARGET   ${triple})
	set(CMAKE_CXX_COMPILER_TARGET ${triple})

	set(CMAKE_AR                  ${TOOLCHAIN_PREFIX}ar${CMAKE_EXECUTABLE_SUFFIX})
	set(CMAKE_ASM_COMPILER        ${TOOLCHAIN_PREFIX}gcc${CMAKE_EXECUTABLE_SUFFIX})
	set(CMAKE_C_COMPILER          ${TOOLCHAIN_PREFIX}gcc${CMAKE_EXECUTABLE_SUFFIX})
	set(CMAKE_CXX_COMPILER        ${TOOLCHAIN_PREFIX}g++${CMAKE_EXECUTABLE_SUFFIX})
	set(CMAKE_LINKER              ${TOOLCHAIN_PREFIX}ld${CMAKE_EXECUTABLE_SUFFIX})
	set(CMAKE_OBJCOPY             ${TOOLCHAIN_PREFIX}objcopy${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
	set(CMAKE_RANLIB              ${TOOLCHAIN_PREFIX}ranlib${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
	set(CMAKE_SIZE                ${TOOLCHAIN_PREFIX}size${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
	set(CMAKE_STRIP               ${TOOLCHAIN_PREFIX}strip${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")

	# -lmsvcrt-os added for symbol '_setjmp' in llvm+clang-14-mingw
	set(CMAKE_CXX_STANDARD_LIBRARIES "-lmsvcrt-os -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32" CACHE STRING "")
	set(CMAKE_C_STANDARD_LIBRARIES "-lmsvcrt-os -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32" CACHE STRING "")

	set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
	set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
	set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
	set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

## Create the cross-compile emulator wrapper around wine

Create a file `/tmp/x86_64-w64-mingw32.cross-emu.sh` with the following
contents

	#!/bin/sh

	exec env \
		WINEDEBUG=fixme-all,err-all \
		WINEPATH='/usr/x86_64-w64-mingw32/usr/bin;/usr/x86_64-w64-mingw32/usr/lib;/usr/lib/gcc/x86_64-w64-mingw32/10-posix' \
		wine64 "$@"

and set it executable with

	chmod +x /tmp/x86_64-w64-mingw32.cross-emu.sh

## Compiling Boolector (optional)

Note, this is experimental, since upstream only tested the win32 thread model:
<https://github.com/Boolector/boolector/blob/master/COMPILING_WINDOWS.md#mingw>

	git clone --depth 1 -b 3.2.3 https://github.com/Boolector/boolector.git &&
	cd boolector &&
	env \
		CC=x86_64-w64-mingw32-gcc \
		CXX=x86_64-w64-mingw32-g++ \
		CPPFLAGS=-DNDEBUG\ -DMS_WIN64\ -DNGETRUSAGE \
		AR=x86_64-w64-mingw32-ar \
		RANLIB=x86_64-w64-mingw32-ranlib \
		contrib/setup-picosat.sh &&
	env \
		CC=x86_64-w64-mingw32-gcc \
		CXX=x86_64-w64-mingw32-g++ \
		CPPFLAGS=-DNDEBUG\ -DMS_WIN64\ -DNGETRUSAGE \
		AR=x86_64-w64-mingw32-ar \
		RANLIB=x86_64-w64-mingw32-ranlib \
		contrib/setup-btor2tools.sh &&
	mkdir build &&
	cd build &&
	cmake .. \
		-DPYTHON=OFF \
		-DIS_WINDOWS_BUILD=1 \
		-GNinja \
		--toolchain /tmp/x86_64-w64-mingw32.toolchain.cmake \
		-DCMAKE_INSTALL_PREFIX=$PWD/installed &&
	ninja install &&
	cd ../.. &&
	BOOLECTOR_OPT=-DBoolector_DIR=$PWD/boolector/build/installed/lib/cmake/Boolector

## Compiling Boost

Boost uses a program called `b2` for building. In the prerequisites we've
installed it, but the version needs to match the one installed on the system.
That's why we're using v1.74.

	cat > /tmp/boost-user-config.jam << EOF &&
	using gcc : 10.3 : x86_64-w64-mingw32-g++ --sysroot=/usr/x86_64-w64-mingw32 : <cflags>" -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -DMS_WIN64 -O3 -DNDEBUG -DNGETRUSAGE -std=gnu11" <cxxflags>" -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -DMS_WIN64 -O3 -DNDEBUG -DNGETRUSAGE -std=gnu++11 -std=c++17" <linkflags>"" <archiver>"x86_64-w64-mingw32-ar" <ranlib>"x86_64-w64-mingw32-ranlib" ;
	EOF
	wget https://boostorg.jfrog.io/artifactory/main/release/1.74.0/source/boost_1_74_0.tar.bz2 &&
	tar xfj boost_1_74_0.tar.bz2 &&
	cd boost_1_74_0 &&
	b2 \
		--user-config=/tmp/boost-user-config.jam \
		--without-python \
		--prefix=$PWD/installed \
		-j`nproc` -q -d+2 pch=off --disable-icu boost.locale.icu=off \
		--without-mpi --without-context --without-coroutine \
		--without-fiber --without-stacktrace --layout=system \
		threading=multi link=static \
		-sNO_BZIP2=1 -sNO_LZMA=1 -sNO_ZLIB=0 -sNO_ZSTD=1 \
		install &&
	cd .. &&
	BOOST_OPT=-DBoost_INCLUDE_DIR=$PWD/boost_1_74_0/installed/include &&
	BOOST_OPT+=\ -DBoost_LIBRARY_DIR=$PWD/boost_1_74_0/installed/lib

## Compiling ESBMC

	mkdir esbmc/build &&
	cd esbmc/build &&
	cmake .. \
		--toolchain /tmp/x86_64-w64-mingw32.toolchain.cmake \
		-DBUILD_STATIC=ON \
		-DBUILD_TESTING=OFF \
		-GNinja \
		-DC2GOTO_SYSROOT=/usr/x86_64-w64-mingw32 \
		$LLVM_OPT \
		$CLANG_OPT \
		$Z3_OPT \
		$BOOLECTOR_OPT \
		$BOOST_OPT \
		-DZLIB_LIBRARY=/usr/x86_64-w64-mingw32/lib/zlib1.dll \
		-DZLIB_INCLUDE_DIR=/usr/x86_64-w64-mingw32/include \
		-DCMAKE_INSTALL_PREFIX=$PWD/installed \
		-DCMAKE_BUILD_TYPE:STRING=Release &&
	ninja install &&
	cd ../..
