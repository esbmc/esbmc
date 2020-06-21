#/bin/bash

export USE_CCACHE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_ANALYTICS=1
export ROOT_DIR=`pwd`
export NINJA_STATUS_SLEEP=2000


travis_before_install() {
    # Here should go changes needed in the repo or system before continuing e.g git submodules
    # ESBMC does not have any for now.
    echo "Configuring repository"

    if [ "$TRAVIS_OS_NAME" = osx ]; then
        # https://docs.travis-ci.com/user/caching/#ccache-on-macos
        export PATH="/usr/local/opt/ccache/libexec:$PATH"
    fi
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
    if [ ! -z "$HOME/boolector-3.2.0" ]; then
        git clone https://github.com/boolector/boolector && cd boolector && git reset --hard 3.2.0 && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./configure.sh --prefix $HOME/boolector-3.2.0 && cd build && make -s -j4 && make install
        cd $ROOT_DIR  
    else
        echo "Boolector cache hit"
    fi
    
}

travis_script() {
    # Compile ESBMC
    if [ "$TRAVIS_OS_NAME" = osx ]; then
        mkdir build
        cd build
        cmake .. -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DBUILD_STATIC=On -DClang_DIR=$PWD/../clang9 -DLLVM_DIR=$PWD/../clang9 -DBoolector_DIR=$HOME/boolector-3.2.0 -DC2GOTO_INCLUDE_DIR=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/ -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../release || echo "cmake warning"
        make -s -j4
    else
        mkdir build
        cd build
        cmake .. -GNinja -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DBUILD_STATIC=On -DClang_DIR=$PWD/../clang9 -DLLVM_DIR=$PWD/../clang9 -DBoolector_DIR=$HOME/boolector-3.2.0 -DCMAKE_INSTALL_PREFIX:PATH=$HOME/release
        ninja
    fi
    
}

travis_after_success() {
    ccache -s
}

set -e
set -x

$1;
