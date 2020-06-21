#/bin/bash

export USE_CCACHE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_ANALYTICS=1
export ROOT_DIR=`pwd`

download_extract() {
    aria2c -x 16 $1 -o $2
    tar -xf $2
}

# This is used for the Android NDK.
download_extract_zip() {
    aria2c --file-allocation=none --timeout=120 --retry-wait=5 --max-tries=20 -Z -c $1 -o $2
    # This resumes the download, in case it failed.
    aria2c --file-allocation=none --timeout=120 --retry-wait=5 --max-tries=20 -Z -c $1 -o $2

    unzip $2 2>&1 | pv > /dev/null
}


travis_before_install() {
    # Here should go changes needed in the repo before continuing e.g git submodules
    # ESBMC does not have any for now.
}

travis_install() {
    # Here are dependencies that were not installed by the addons e.g solvers, llvm

    # LLVM
    if [ "$TRAVIS_OS_NAME" = osx ]; then
        wget https://github.com/llvm/llvm-project/releases/download/llvmorg-9.0.1/clang+llvm-9.0.1-x86_64-apple-darwin.tar.xz
        tar xf clang+llvm-9.0.1-x86_64-apple-darwin.tar.xz && mv clang+llvm-9.0.1-x86_64-apple-darwin clang9
    else
        wget http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
        tar xf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && mv clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04 clang9
    fi

    # Boolector
    git clone https://github.com/boolector/boolector && cd boolector && git reset --hard 3.2.0 && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./configure.sh --prefix $PWD/../boolector-release && cd build && make -j4 && make install
    cd $ROOT_DIR  
}

travis_script() {
    # Compile ESBMC
    if [ "$TRAVIS_OS_NAME" = osx ]; then
        mkdir build
        cd build
        cmake .. -GNinja -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DBUILD_STATIC=On -DClang_DIR=$PWD/../clang9 -DLLVM_DIR=$PWD/../clang9 -DBoolector_DIR=$PWD/../boolector-release -DC2GOTO_INCLUDE_DIR=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/ -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../release
        ninja
    else
        mkdir build
        cd build
        cmake .. -GNinja -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DBUILD_STATIC=On -DClang_DIR=$PWD/../clang9 -DLLVM_DIR=$PWD/../clang9 -DBoolector_DIR=$PWD/../boolector-release -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../release
        ninja
    fi
    
}

travis_after_success() {
    ccache -s
}

set -e
set -x

$1;
