#!/bin/sh

# Set arguments that should be available for every OS
BASE_ARGS="\
    -DDOWNLOAD_DEPENDENCIES=On \
    -GNinja \
    -DENABLE_CSMITH=On \
    -DBUILD_TESTING=On \
    -DENABLE_REGRESSION=On \
    -DENABLE_SOLIDITY_FRONTEND=On \
    -DENABLE_JIMPLE_FRONTEND=On \
    -DCMAKE_INSTALL_PREFIX:PATH=$PWD/release \
"
SOLVER_FLAGS="\
    -DENABLE_BOOLECTOR=On \
    -DENABLE_YICES=Off \
    -DENABLE_CVC4=OFF \
    -DENABLE_BITWUZLA=On \
    -DENABLE_GOTO_CONTRACTOR=OFF \
"
COMPILER_ARGS=''

STATIC=
CLANG_VERSION=11

error() {
    echo "$*" >&2
    exit 1
}

# Ubuntu setup (pre-config)
ubuntu_setup () {
    # Tested on ubuntu 22.04
    PKGS="\
        clang-$CLANG_VERSION clang-tidy-$CLANG_VERSION \
        python-is-python3 csmith python3 \
        git ccache unzip wget curl libcsmith-dev gperf \
        libgmp-dev cmake bison flex g++-multilib linux-libc-dev \
        libboost-all-dev ninja-build python3-setuptools \
        libtinfo-dev pkg-config python3-pip python3-toml \
        openjdk-11-jdk \
    "
    if [ -z "$STATIC" ]; then STATIC=ON; fi
    if [ $STATIC = OFF ]; then
        PKGS="$PKGS \
            llvm-$CLANG_VERSION-dev \
            libclang-$CLANG_VERSION-dev \
            libclang-cpp$CLANG_VERSION-dev \
            libz3-dev \
        "
        BASE_ARGS="$BASE_ARGS \
            -DClang_DIR=/usr/lib/cmake/clang-$CLANG_VERSION \
            -DLLVM_DIR=/usr/lib/llvm-$CLANG_VERSION/lib/cmake/llvm \
            -DZ3_DIR=/usr \
        "
        echo "Configuring shared Ubuntu build with Clang v$CLANG_VERSION"
    else
        echo "Configuring static Ubuntu build"
    fi
    sudo apt-get update &&
    sudo apt-get install -y $PKGS &&
    echo "Installing Python dependencies" &&
    pip3 install --user meson ast2json &&
    meson --version &&
    BASE_ARGS="$BASE_ARGS \
        -DENABLE_OLD_FRONTEND=On \
        -DENABLE_PYTHON_FRONTEND=On \
        -DBUILD_STATIC=$STATIC \
    " &&
    SOLVER_FLAGS="$SOLVER_FLAGS \
        -DENABLE_Z3=ON \
    " &&
    # Hack: Boolector might fail to download some dependencies using curl (maybe we should patch it?)
    # curl: (60) SSL: no alternative certificate subject name matches target host name 'codeload.github.com'
    # As a unsafe workaround... we can just tell curl to be unsafe
    echo "insecure" > $HOME/.curlrc
}

ubuntu_post_setup () {
  echo "No further steps needed for ubuntu"
}

# macOS setup (pre-config)
macos_setup () {
    echo "Configuring macOS build"
    if [ -z "$STATIC" ]; then STATIC=OFF; fi
    if [ $STATIC = ON ]; then
        error "static macOS build is currently not supported"
    fi
    brew install z3 gmp csmith cmake boost ninja python3 automake bison flex &&
    pip3 install PySMT toml ast2json &&
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-apple-darwin.tar.xz &&
    tar xf clang+llvm-11.0.0-x86_64-apple-darwin.tar.xz &&
    mv clang+llvm-11.0.0-x86_64-apple-darwin ../clang &&
    BASE_ARGS="$BASE_ARGS \
        -DENABLE_WERROR=Off \
        -DBUILD_STATIC=$STATIC \
        -DClang_DIR=$PWD/../clang \
        -DLLVM_DIR=$PWD/../clang \
        -DC2GOTO_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
    "
}

# macOS needs an extra step before testing
macos_post_setup () {
  chmod +x build/macos-wrapper.sh
}

# Create build directory
mkdir build && cd build || exit $?

# Detect the platform ($OSTYPE was not working on github actions for ubuntu)
# Note: Linux here means Ubuntu, this will mostly not work anywhere else.
OS=`uname`

# Setup build flags (release, debug, sanitizer, ...)
while getopts b:s:e:r:dS:c: flag
do
    case "${flag}" in
        b) BASE_ARGS="$BASE_ARGS -DCMAKE_BUILD_TYPE=${OPTARG}";;
        s) BASE_ARGS="$BASE_ARGS -DSANITIZER_TYPE=${OPTARG}"
           COMPILER_ARGS="$COMPILER_ARGS CC=clang CXX=clang++";;
        e) BASE_ARGS="$BASE_ARGS -DENABLE_WERROR=${OPTARG}";;
        r) BASE_ARGS="$BASE_ARGS -DBENCHBRINGUP=${OPTARG}" ;;
        d) set -x; export ESBMC_OPTS='--verbosity 9' ;;
        S) STATIC=$OPTARG ;; # should be capital ON or OFF
        c) CLANG_VERSION=$OPTARG ;; # LLVM/Clang major version
    esac
done

case $OS in
  'Linux')
    ubuntu_setup
    ;;
  'Darwin')
    macos_setup
    ;;
  *) echo "Unsupported OS $OSTYPE" ; exit 1; ;;
esac || exit $?

# Configure ESBMC
printf "Running CMake:"
printf " '%s'" $COMPILER_ARGS cmake .. $BASE_ARGS $SOLVER_FLAGS
echo
$COMPILER_ARGS cmake .. $BASE_ARGS $SOLVER_FLAGS &&
# Compile ESBMC
cmake --build . && ninja install || exit $?

case $OS in
  'Linux')
    ubuntu_post_setup
    ;;
  'Darwin')
    macos_post_setup
    ;;
  *) echo "Unsupported OS $OSTYPE" ; exit 1; ;;
esac || exit $?
