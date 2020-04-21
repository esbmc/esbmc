# ESBMC Build Guide

This is a guide on how to build ESBMC and its supported solvers.

It has been tested with Ubuntu 19.10 and macOS Catalina, but the steps are mostly the same for other Linux and macOS distributions.

Before starting, note that ESBMC is mainly distributed under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), so please make sure to read it carefully.

## Installing Dependencies

We need to install some dependencies before moving into next steps.

All of them are listed in the following installation command:

```
Linux:
sudo apt-get update && sudo apt-get install gperf libgmp-dev cmake bison curl flex gcc-multilib linux-libc-dev libboost-all-dev libtinfo-dev ninja-build python3-setuptools

macOS:
brew install gmp cmake boost ninja python3 automake && pip3 install PySMT
```

Note that they are listed with their name in Debian/Ubuntu, but they can be found in other distributions as well.

## Cloning ESBMC Source Code

ESBMC source code is publicly available in [Github](https://github.com/esbmc/esbmc).

You can get the latest version using the following __git__ command:

```
git clone https://github.com/esbmc/esbmc
```

## Preparing Clang 9

ESBMC uses [__clang__](https://clang.llvm.org/) in its front-end. It currently supports version 9.0.0.

First, we need to download the package. It can be performed using the following __wget__ command:

```
Linux:
wget http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz

macOS:
wget http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-darwin-apple.tar.xz
```

Then, we need to extract the package. You can use the following __tar__ command:

```
Linux:
tar xf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && mv clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04 clang9

macOS:
tar xf clang+llvm-9.0.0-x86_64-darwin-apple.tar.xz && mv clang+llvm-9.0.0-x86_64-darwin-apple clang9
```

## Setting Up Solvers

ESBMC relies on SMT solvers to reason about formulae in its back-end.

Currently we support the following solvers: __Boolector__, __CVC4__, __MathSAT__, __Yices 2__, and __Z3__.

Since this guide focuses primarily on ESBMC build, we will only cover the steps needed by it.

### Setting Up Boolector

We have wrapped the entire build and setup of Boolector in the following command:

```
git clone https://github.com/boolector/boolector && cd boolector && git reset --hard 3.2.0 && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./configure.sh --prefix $PWD/../boolector-release && cd build && make -j9 && make install
```

If you need more details on Boolector, please refer to [its Github](https://github.com/Boolector/boolector).

### Setting Up CVC4 (Linux Only)

We have wrapped the entire build and setup of CVC4 in the following command:

```
wget https://github.com/CVC4/CVC4/archive/1.7.tar.gz && tar xf 1.7.tar.gz && rm 1.7.tar.gz && cd CVC4-1.7 && ./contrib/get-antlr-3.4 && ./configure.sh --optimized --prefix=../cvc4 --static --no-static-binary && cd build && make -j8 && make install
```

If you need more details on CVC4, please refer to [its Github](https://github.com/CVC4/CVC4).

### Setting Up MathSAT

We have wrapped the entire build and setup of MathSAT in the following command:

```
Linux:
wget http://mathsat.fbk.eu/download.php?file=mathsat-5.5.4-linux-x86_64.tar.gz -O mathsat.tar.gz && tar xf mathsat.tar.gz && mv mathsat-5.5.4-linux-x86_64 mathsat

macOS:
wget http://mathsat.fbk.eu/download.php?file=mathsat-5.5.4-darwin-libcxx-x86_64.tar.gz -O mathsat.tar.gz && tar xf mathsat.tar.gz && mv mathsat-5.5.4-darwin-libcxx-x86_64 mathsat
```

In macOS, the following command is required:

```
ln -s /usr/local/include/gmp.h mathsat/include/gmp.h
```

If you need more details on MathSAT, please refer to [its webpage](https://mathsat.fbk.eu).

### Setting Up Yices

First, we need to setup and build [GMP library](https://gmplib.org), by entering the following command (Linux only):

```
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz && tar xf gmp-6.1.2.tar.xz && rm gmp-6.1.2.tar.xz && cd gmp-6.1.2 && ./configure --prefix $PWD/../gmp --disable-shared ABI=64 CFLAGS=-fPIC CPPFLAGS=-DPIC && make -j4 && make install
```

Then, we are able build and setup Yices 2 using the following command:

```
Linux:
git clone https://github.com/SRI-CSL/yices2.git && cd yices2 && git checkout Yices-2.6.1 && autoreconf -fi && ./configure --prefix $PWD/../yices --with-static-gmp=$PWD/../gmp/lib/libgmp.a && make -j9 && make static-lib && make install && cp ./build/x86_64-pc-linux-gnu-release/static_lib/libyices.a ../yices/lib

macOS:
git clone https://github.com/SRI-CSL/yices2.git && cd yices2 && git checkout Yices-2.6.1 && autoreconf -fi && ./configure --prefix $PWD/../yices && make -j9 && make static-lib && make install && cp ./build/x86_64-apple-darwin*release/static_lib/libyices.a ../yices/lib
```

If you need more details on Yices 2, please refer to [its Github](https://github.com/SRI-CSL/yices2).

### Setting Up Z3

We have wrapped the entire build and setup of Z3 in the following command:

```
Linux:
wget https://github.com/Z3Prover/z3/releases/download/z3-4.8.4/z3-4.8.4.d6df51951f4c-x64-ubuntu-16.04.zip && unzip z3-4.8.4.d6df51951f4c-x64-ubuntu-16.04.zip && mv z3-4.8.4.d6df51951f4c-x64-ubuntu-16.04 z3

macOS:
brew install z3 
```

If you need more details on Z3, please refer to [its Github](https://github.com/Z3Prover/z3).

## Building ESBMC

Now we are ready to build ESBMC.

First, we need to setup __cmake__, by using the following command:

```
Linux:
cd esbmc && mkdir build && cd build && cmake .. -GNinja -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DClang_DIR=$PWD/../../clang9 -DLLVM_DIR=$PWD/../../clang9 -DBUILD_STATIC=On -DBoolector_DIR=$PWD/../../boolector-release -DZ3_DIR=$PWD/../../z3 -DENABLE_MATHSAT=ON -DMathsat_DIR=$PWD/../../mathsat -DENABLE_YICES=On -DYices_DIR=$PWD/../../yices -DCVC4_DIR=$PWD/../../cvc4 -DGMP_DIR=$PWD/../../gmp -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../release

macOS:
mkdir build && cd build && cmake .. -GNinja -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DBUILD_STATIC=On -DClang_DIR=$PWD/../../clang9 -DLLVM_DIR=$PWD/../../clang9 -DBoolector_DIR=$PWD/../../boolector-release -DZ3_DIR=$PWD/../../z3 -DENABLE_MATHSAT=On -DMathsat_DIR=$PWD/../../mathsat -DENABLE_YICES=ON -DYices_DIR=$PWD/../../yices -DC2GOTO_INCLUDE_DIR=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/ -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../release
```

Finally, we can trigger the build process, by using the following command:

```
cmake --build . && ninja install
```

Once it is finished, ESBMC should be available in the _release_ folder.
