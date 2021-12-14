# ESBMC Build Guide

This is a guide on how to build ESBMC and its supported solvers.

It has been tested with Ubuntu 20.04.1 and macOS Catalina as well as [Windows 10 WSL](https://docs.microsoft.com/en-us/windows/wsl/about) v1 or v2 [Ubuntu WSL](https://ubuntu.com/wsl), but the steps are mostly the same for other Linux and macOS distributions.

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

## Preparing CHERI Clang 13

CHERI-enabled ESBMC uses [__CHERI clang__](https://github.com/CTSRD-CHERI/llvm-project) in its front-end. It currently supports the release 20210817 based on clang version 13.0.0.

First, we need to download the package. It can be performed using the following __wget__ command:

```
wget https://github.com/CTSRD-CHERI/llvm-project/archive/refs/tags/cheri-rel-20210817.tar.gz
```

Then, we need to extract and build the package. You can use the following commands:

```
tar xf cheri-rel-20210817.tar.gz &&
mkdir clang13 &&
cd llvm-project-cheri-rel-20210817 &&
mkdir build &&
cd build &&
cmake -GNinja -S ../llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_PARALLEL_LINK_JOBS=4 -DLLVM_CCACHE_BUILD=FALSE -DLLVM_INSTALL_BINUTILS_SYMLINKS=TRUE -DLLVM_ENABLE_LIBXML2=FALSE -DLLVM_ENABLE_ZLIB=FORCE_ON -DLLVM_ENABLE_OCAMLDOC=FALSE -DLLVM_ENABLE_BINDINGS=FALSE -DLLVM_INCLUDE_EXAMPLES=FALSE -DLLVM_INCLUDE_DOCS=FALSE -DLLVM_INCLUDE_BENCHMARKS=FALSE -DCLANG_ENABLE_STATIC_ANALYZER=FALSE -DCLANG_ENABLE_ARCMT=FALSE -DLLVM_ENABLE_Z3_SOLVER=FALSE -DLLVM_TOOL_LLVM_MCA_BUILD=FALSE -DLLVM_TOOL_LLVM_EXEGESIS_BUILD=FALSE -DLLVM_TOOL_LLVM_RC_BUILD=FALSE -DLLVM_ENABLE_LLD=TRUE -DLLVM_OPTIMIZED_TABLEGEN=FALSE -DLLVM_USE_SPLIT_DWARF=TRUE -DLLVM_ENABLE_ASSERTIONS=TRUE '-DLLVM_LIT_ARGS=--max-time 3600 --timeout 300 -s -vv' '-DLLVM_TARGETS_TO_BUILD=AArch64;ARM;Mips;RISCV;X86;host' -DENABLE_EXPERIMENTAL_NEW_PASS_MANAGER=FALSE -DCLANG_ROUND_TRIP_CC1_ARGS=FALSE '-DLLVM_ENABLE_PROJECTS=llvm;clang;lld' -DCMAKE_INSTALL_PREFIX=../../clang13 -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_ASM_COMPILER=/usr/bin/cc -DCMAKE_BUILD_RPATH_USE_ORIGIN=TRUE
ninja &&
ninja install &&
cd ../..
```

## Setting Up Solvers

ESBMC relies on SMT solvers to reason about formulae in its back-end.

Currently we support the following solvers: __Bitwuzla__, __Boolector__, __CVC4__, __MathSAT__, __Yices 2__, and __Z3__.

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

If you are using macOS and have installed z3 using brew, you'll need to copy the entire z3 directory from the place where Homebrew keeps packages to your workspace:

```
cp -rp $(brew info z3 | egrep "/usr[/a-zA-Z\.0-9]+ " -o) z3
```

### Setting Up Bitwuzla

We have wrapped the entire build and setup of Bitwuzla in the following command:

```
Linux:
git clone --depth=1 --branch=smtcomp-2021 https://github.com/bitwuzla/bitwuzla.git && cd bitwuzla && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./contrib/setup-symfpu.sh && ./configure.sh --prefix $PWD/../bitwuzla-release && cd build && cmake -DGMP_INCLUDE_DIR=$PWD/../../gmp/include -DGMP_LIBRARIES=$PWD/../../gmp/lib/libgmp.a -DONLY_LINGELING=ON ../ && make -j8 && make install && cd .. && cd ..

macOS:
git clone --depth=1 --branch=smtcomp-2021 https://github.com/bitwuzla/bitwuzla.git && cd bitwuzla && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./contrib/setup-symfpu.sh && ./configure.sh --prefix $PWD/../bitwuzla-release && cd build && cmake -DONLY_LINGELING=ON ../ && make -j8 && make install && cd .. && cd ..
```

If you need more details on Bitwuzla, please refer to [its Github](https://github.com/bitwuzla/bitwuzla).

Before proceeding to the next section, make sure you have clang, LLVM and all the solvers ready in your workspace:

```
clang and LLVM:
<path_to_your_project>/ESBMC_Project/clang13

Boolector:
<path_to_your_project>/ESBMC_Project/boolector

CVC4: (Linux only)
<path_to_your_project>/ESBMC_Project/CVC4

MathSAT:
<path_to_your_project>/ESBMC_Project/mathsat

Yices:
<path_to_your_project>/ESBMC_Project/yices

Z3:
<path_to_your_project>/ESBMC_Project/z3

Bitwuzla:
<path_to_your_project>/ESBMC_Project/bitwuzla
```

The above paths will be used in ESBMC's build command in the next section.

If you are using macOS, make sure you have Xcode.app in your Applications directory by checking:

```
ls /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/
```

If no such directory, please go to App Store and install Xcode. If you do not have the full version of Xcode, you will need to replace the flag `C2GOTO` with:

```
-DC2GOTO_INCLUDE_DIR=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include
```

## Building ESBMC

Now we are ready to build ESBMC. Please note that we describe the same build option used in our CI/CD. If you want to all available _cmake_ options, refer to our [Options.cmake file](https://github.com/esbmc/esbmc/blob/master/scripts/cmake/Options.cmake).

First, we need to setup __cmake__, by using the following command in ESBMC_Project directory you just created:

```
Linux:
cd esbmc && mkdir build && cd build && cmake .. -GNinja -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DClang_DIR=$PWD/../../clang13 -DLLVM_DIR=$PWD/../../clang13 -DBUILD_STATIC=On -DBoolector_DIR=$PWD/../../boolector-release -DZ3_DIR=$PWD/../../z3 -DENABLE_MATHSAT=ON -DMathsat_DIR=$PWD/../../mathsat -DENABLE_YICES=On -DYices_DIR=$PWD/../../yices -DCVC4_DIR=$PWD/../../cvc4 -DGMP_DIR=$PWD/../../gmp -DBitwuzla_DIR=$PWD/../../bitwuzla-release -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../../release

macOS:
cd esbmc && mkdir build && cd build && cmake .. -GNinja -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DBUILD_STATIC=On -DClang_DIR=$PWD/../../clang13 -DLLVM_DIR=$PWD/../../clang13 -DBoolector_DIR=$PWD/../../boolector-release -DZ3_DIR=$PWD/../../z3 -DENABLE_MATHSAT=On -DMathsat_DIR=$PWD/../../mathsat -DENABLE_YICES=ON -DYices_DIR=$PWD/../../yices -DC2GOTO_INCLUDE_DIR=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/ -DBitwuzla_DIR=$PWD/../../bitwuzla-release -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../../release
```

Finally, we can trigger the build process, by using the following command:

```
cmake --build . && ninja install
```

Once it is finished, ESBMC should be available in the _release_ folder.


## Setting up the sysroot for CHERI clang

CHERI clang is used as frontend by CHERI-enabled ESBMC.
Since CHERI support is available only for a few platforms, verifying CHERI-C programs that use header files from the C standard library will require a setup of a C standard library for one of these platforms.
CHERI-enabled ESBMC defaults to the platform mips64-unknown-linux-gnu and
expects the corresponding sysroot to be accessible at `/usr/mips64-unknown-linux-gnu`.

In case you are running CHERI-enabled ESBMC on such a platform, the system can
directly be used, only a symlink has to be set up in case it does not yet exist:
```
test -e /usr/mips64-unknown-linux-gnu || su -c 'ln -s . /usr/mips64-unknown-linux-gnu'
```

Otherwise, to install the library and its headers, on Ubuntu run
```
sudo apt install libc6-dev-mips64-cross &&
sudo ln -s mips64-linux-gnuabi64 /usr/mips64-unknown-linux-gnu
```
and on Gentoo execute (as root)
```
emerge -n crossdev && crossdev -t mips64-unknown-linux-gnu --stage3 -A "n64 n32"
```
