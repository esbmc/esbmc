#!/bin/sh

mkdir build && cd build

BASE_ARGS="-DDOWNLOAD_DEPENDENCIES=On -GNinja -DENABLE_CSMITH=On -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DENABLE_SOLIDITY_FRONTEND=On -DENABLE_JIMPLE_FRONTEND=On -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../release"
SOLVER_FLAGS="-DENABLE_BOOLECTOR=On -DENABLE_Z3=On -DENABLE_YICES=Off -DENABLE_CVC4=OFF  -DENABLE_BITWUZLA=OFF -DENABLE_GOTO_CONTRACTOR=OFF"

ubuntu_setup () {
    sudo apt-get update && sudo apt-get install -y clang clang-tidy python-is-python3 csmith python3 git ccache unzip wget curl libcsmith-dev gperf libgmp-dev cmake bison flex gcc-multilib linux-libc-dev libboost-all-dev ninja-build python3-setuptools libtinfo-dev pkg-config python3-pip python3-toml python-is-python3 openjdk-11-jdk
    BASE_ARGS="$BASE_ARGS -DENABLE_OLD_FRONTEND=On  -DENABLE_WERROR=ON -DDOWNLOAD_DEPENDENCIES=On -DBUILD_STATIC=On"
}

macos_setup () {
   brew install z3 gmp csmith cmake boost ninja python3 automake bison flex && pip3 install PySMT toml
   wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-apple-darwin.tar.xz
   tar xf clang+llvm-11.0.0-x86_64-apple-darwin.tar.xz && mv clang+llvm-11.0.0-x86_64-apple-darwin ../clang
   BASE_ARGS="$BASE_ARGS -DENABLE_WERROR=Off -DBUILD_STATIC=Off -DClang_DIR=$PWD/../clang -DLLVM_DIR=$PWD/../clang -DBoolector_DIR=$PWD/../boolector-release -DZ3_DIR=$PWD/../z3 -DC2GOTO_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
}

# TODO: windows? win32

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Setting up Ubuntu mode"
    # TODO: we should support other Linux as well
    ubuntu_setup
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Setting up MacOS mode"
    # TODO: M1/M2 mode?
    macos_setup
else
    echo "Unsupported OS $OSTYPE" ; exit 1;
fi


COMPILER_ARGS=''
while getopts b:s: flag
do
    case "${flag}" in
        b) BASE_ARGS="$BASE_ARGS -DCMAKE_BUILD_TYPE=${OPTARG}";;
        s) BASE_ARGS="$BASE_ARGS -DSANITIZER_TYPE=${OPTARG}"
           COMPILER_ARGS="CC=clang CXX=clang++";;
    esac
done

$COMPILER_ARGS cmake .. $BASE_ARGS
cmake --build . && ninja install
