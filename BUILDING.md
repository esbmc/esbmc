# ESBMC Build Guide

This is a guide on how to build ESBMC and its supported solvers.

It has been tested with Ubuntu 20.04.1 and macOS Catalina, but the steps are mostly the same for other Linux and macOS distributions.

It is recommended that the RAM should be 6 GB at least.

Before starting, note that ESBMC is mainly distributed under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), so please make sure to read it carefully.

## Installing Dependencies

We need to install some dependencies before moving into next steps.

All of them are listed in the following installation command:

```
Linux:
sudo apt-get update && sudo apt-get install build-essential git gperf libgmp-dev cmake bison curl flex gcc-multilib linux-libc-dev libboost-all-dev libtinfo-dev ninja-build python3-setuptools unzip wget python3-pip openjdk-8-jre

macOS:
brew install gmp cmake boost ninja python3 automake && pip3 install PySMT
```

Note that they are listed with their name in Debian/Ubuntu, but they can be found in other distributions as well.

## Cloning ESBMC Source Code

ESBMC source code is publicly available in [Github](https://github.com/esbmc/esbmc).

Before Cloning ESBMC Source Code, it is better to make a directory to contain the whole project in “ESBMC_Project”.

You can get the latest version using the following __git__ command:

```
mkdir ESBMC_Project && cd ESBMC_Project && git clone https://github.com/esbmc/esbmc 
```

## Preparing Clang 11

ESBMC uses [__clang__](https://clang.llvm.org/) in its front-end. It currently supports version 11.0.0.

First, we need to download the package. It can be performed using the following __wget__ command:

```
Linux:
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz

macOS:
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-apple-darwin.tar.xz
```

Then, we need to extract the package. You can use the following __tar__ command:

```
Linux:
tar xJf clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz && mv clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04 clang11

macOS:
tar xJf clang+llvm-11.0.0-x86_64-apple-darwin.tar.xz && mv clang+llvm-11.0.0-x86_64-darwin-apple clang11
```

## Setting Up Solvers

ESBMC relies on SMT solvers to reason about formulae in its back-end.

Currently we support the following solvers: __Boolector__, __CVC4__, __MathSAT__, __Yices 2__, and __Z3__.

Since this guide focuses primarily on ESBMC build, we will only cover the steps needed by it.

### Setting Up Boolector

We have wrapped the entire build and setup of Boolector in the following command:

```
git clone --depth=1 --branch=3.2.1 https://github.com/boolector/boolector && cd boolector && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./configure.sh --prefix $PWD/../boolector-release && cd build && make -j9 && make install && cd .. && cd ..
```

If you need more details on Boolector, please refer to [its Github](https://github.com/Boolector/boolector).

### Setting Up CVC4 (Linux Only)

We have wrapped the entire build and setup of CVC4 in the following command:

```
pip3 install toml && git clone https://github.com/CVC4/CVC4.git && cd CVC4 && git reset --hard b826fc8ae95fc && ./contrib/get-antlr-3.4 && ./configure.sh --optimized --prefix=../cvc4 --static --no-static-binary && cd build && make -j4 && make install && cd .. && cd ..
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
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz && tar xf gmp-6.1.2.tar.xz && rm gmp-6.1.2.tar.xz && cd gmp-6.1.2 && ./configure --prefix $PWD/../gmp --disable-shared ABI=64 CFLAGS=-fPIC CPPFLAGS=-DPIC && make -j4 && make install && cd ..
```

Then, we are able build and setup Yices 2 using the following command:

```
Linux:
git clone https://github.com/SRI-CSL/yices2.git && cd yices2 && git checkout Yices-2.6.1 && autoreconf -fi && ./configure --prefix $PWD/../yices --with-static-gmp=$PWD/../gmp/lib/libgmp.a && make -j9 && make static-lib && make install && cp ./build/x86_64-pc-linux-gnu-release/static_lib/libyices.a ../yices/lib && cd ..

macOS:
git clone https://github.com/SRI-CSL/yices2.git && cd yices2 && git checkout Yices-2.6.1 && autoreconf -fi && ./configure --prefix $PWD/../yices && make -j9 && make static-lib && make install && cp ./build/x86_64-apple-darwin*release/static_lib/libyices.a ../yices/lib
```

If you need more details on Yices 2, please refer to [its Github](https://github.com/SRI-CSL/yices2).

### Setting Up Z3

We have wrapped the entire build and setup of Z3 in the following command:

```
Linux:
wget https://github.com/Z3Prover/z3/releases/download/z3-4.8.9/z3-4.8.9-x64-ubuntu-16.04.zip && unzip z3-4.8.9-x64-ubuntu-16.04.zip && mv z3-4.8.9-x64-ubuntu-16.04 z3

macOS:
brew install z3
```

If you need more details on Z3, please refer to [its Github](https://github.com/Z3Prover/z3).

## Building ESBMC

Now we are ready to build ESBMC. Please note that we describe the same build option used in our CI/CD. If you want to all available _cmake_ options, refer to our [Options.cmake file](https://github.com/esbmc/esbmc/blob/master/scripts/cmake/Options.cmake).

First, we need to setup __cmake__, by using the following command:

```
Linux:
cd esbmc && mkdir build && cd build && cmake .. -GNinja -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DClang_DIR=$PWD/../../clang11 -DLLVM_DIR=$PWD/../../clang11 -DBUILD_STATIC=On -DBoolector_DIR=$PWD/../../boolector-release -DZ3_DIR=$PWD/../../z3 -DENABLE_MATHSAT=ON -DMathsat_DIR=$PWD/../../mathsat -DENABLE_YICES=On -DYices_DIR=$PWD/../../yices -DCVC4_DIR=$PWD/../../cvc4 -DGMP_DIR=$PWD/../../gmp -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../../release

macOS:
cd esbmc && mkdir build && cd build && cmake .. -GNinja -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DBUILD_STATIC=On -DClang_DIR=$PWD/../../clang11 -DLLVM_DIR=$PWD/../../clang11 -DBoolector_DIR=$PWD/../../boolector-release -DZ3_DIR=$PWD/../../z3 -DENABLE_MATHSAT=On -DMathsat_DIR=$PWD/../../mathsat -DENABLE_YICES=ON -DYices_DIR=$PWD/../../yices -DC2GOTO_INCLUDE_DIR=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/ -DCMAKE_INSTALL_PREFIX:PATH=$PWD/..//..release
```

Finally, we can trigger the build process, by using the following command:

```
cmake --build . && ninja install
```

Once it is finished, ESBMC should be available in the _release_ folder.
