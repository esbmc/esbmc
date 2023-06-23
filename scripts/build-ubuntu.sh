#!/bin/sh

sudo apt-get update && sudo apt-get install -y clang clang-tidy python-is-python3 csmith python3 git ccache unzip wget curl libcsmith-dev gperf libgmp-dev cmake bison flex gcc-multilib linux-libc-dev libboost-all-dev ninja-build python3-setuptools libtinfo-dev pkg-config python3-pip python3-toml python-is-python3 openjdk-11-jdk
mkdir build && cd build

BASE_ARGS='-DENABLE_SOLIDITY_FRONTEND=On -DENABLE_OLD_FRONTEND=On -DENABLE_JIMPLE_FRONTEND=On -GNinja -DENABLE_WERROR=ON -DENABLE_CSMITH=On -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DDOWNLOAD_DEPENDENCIES=On -DBUILD_STATIC=On -DENABLE_BOOLECTOR=On -DENABLE_YICES=Off -DENABLE_CVC4=OFF -DENABLE_Z3=On -DENABLE_BITWUZLA=On -DENABLE_GOTO_CONTRACTOR=OFF -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../release'
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
