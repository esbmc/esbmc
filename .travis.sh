#/bin/bash

export USE_CCACHE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_ANALYTICS=1
export ROOT_DIR=`pwd`
export NINJA_STATUS_SLEEP=2000

export ESBMC_CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX:PATH=$HOME/release -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DBUILD_STATIC=On"


travis_before_install() {
    # Here should go changes needed in the repo or system before continuing e.g git submodules, envvars
    # ESBMC does not have any for now.
    echo "Configuring repository"

    if [ "$TRAVIS_OS_NAME" = osx ]; then
        # https://docs.travis-ci.com/user/caching/#ccache-on-macos
        export PATH="/usr/local/opt/ccache/libexec:$PATH"
        export ESBMC_CMAKE_ARGS="$ESBMC_CMAKE_ARGS -DC2GOTO_INCLUDE_DIR=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/"
    fi
}

travis_install() {
    # Here are dependencies that were not installed by the addons e.g solvers, llvm

    # LLVM
    if [ "$TRAVIS_OS_NAME" = osx ]; then
        wget https://github.com/llvm/llvm-project/releases/download/llvmorg-9.0.1/clang+llvm-9.0.1-x86_64-apple-darwin.tar.xz
        tar xf clang+llvm-9.0.1-x86_64-apple-darwin.tar.xz && mv clang+llvm-9.0.1-x86_64-apple-darwin $HOME/clang9
    else
        wget http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
        tar xf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && mv clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04 $HOME/clang9
    fi

    export ESBMC_CMAKE_ARGS="$ESBMC_CMAKE_ARGS  -DLLVM_DIR=$HOME/clang9 -DClang_DIR=$HOME/clang9"

    # Boolector
    if [ ! -z "$HOME/boolector-3.2.0" ]; then
        git clone https://github.com/boolector/boolector && cd boolector && git reset --hard 3.2.0 && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./configure.sh --prefix $HOME/boolector-3.2.0 && cd build && make -s -j4 && make install
        cd $ROOT_DIR
    else
        echo "Boolector cache hit"
    fi
    
    export ESBMC_CMAKE_ARGS="$ESBMC_CMAKE_ARGS -DBoolector_DIR=$HOME/boolector-3.2.0"
}

travis_script() {
    # Compile ESBMC
    mkdir build
    cd build
    if [ "$TRAVIS_OS_NAME" = osx ]; then        
        cmake .. $ESBMC_CMAKE_ARGS || echo "cmake warning"
        make -s -j4
    else
        mkdir build
        cd build
        cmake .. -GNinja $ESBMC_CMAKE_ARGS
        ninja
    fi
    
}

travis_after_success() {
    ccache -s
}

set -e
set -x

$1;
