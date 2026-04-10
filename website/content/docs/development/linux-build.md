---
title: Linux Build
---

This page contains information about a simplified building process for ESBMC on Linux. We consider that the user has the ubuntu-20.04 OS. 

1) In order to install dependencies, the user needs to run the following command: 

`sudo apt-get update && sudo apt-get install gperf libgmp-dev cmake bison flex gcc-multilib linux-libc-dev libboost-all-dev ninja-build python3-setuptools libtinfo-dev build-essential`

2) The user should download Clang 11 as follows:

`wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz --no-check-certificate` 

and then extract it in the `home` directory as 

`tar xf clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz && mv clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04 clang`

3) Install the SMT solver Boolector or Z3 as 

Boolector:

`git clone https://github.com/boolector/boolector && cd boolector && git reset --hard 3.2.0 && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./configure.sh --prefix $PWD/../boolector-release && cd build && make -j9 && make install`

**Note that you should visualize `clang`, `boolector`, and `boolector-release` directories when you type `ls` in the `home` directory.**

Z3:

`wget https://github.com/Z3Prover/z3/releases/download/z3-4.8.9/z3-4.8.9-x64-ubuntu-16.04.zip --no-check-certificate && unzip z3-4.8.9-x64-ubuntu-16.04.zip && mv z3-4.8.9-x64-ubuntu-16.04/ z3`

**Note that you should visualize `clang` and `z3` directories when you type `ls` in the `home` directory.**

4) You can get the latest ESBMC version using the following __git__ command: 

`git clone https://github.com/esbmc/esbmc`

5) Go to the ESBMC directory (using `cd esbmc`) and configure CMake either with Boolector or Z3 using the following command: 

Boolector:

`mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja -DClang_DIR=$PWD/../../clang -DLLVM_DIR=$PWD/../../clang -DBUILD_STATIC=On -DBoolector_DIR=$PWD/../../boolector-release -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../../release`

Z3:

`mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja -DClang_DIR=$PWD/../../clang -DLLVM_DIR=$PWD/../../clang -DBUILD_STATIC=On -DZ3_DIR=$PWD/../../z3 -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../../release`

6) Finally, we can trigger the build process, by using the following command from the `build` directory: 

`cmake --build . && ninja install`

Once it is finished, ESBMC should be available in the `release` folder.
