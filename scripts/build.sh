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
# Must disable old frontend to enable goto contractor. github issue #1110
# https://github.com/esbmc/esbmc/issues/1110.
SOLVER_FLAGS="\
    -DENABLE_BOOLECTOR=On \
    -DENABLE_YICES=Off \
    -DENABLE_CVC4=OFF \
    -DENABLE_BITWUZLA=On \
    -DENABLE_GOTO_CONTRACTOR=On \
    -DACADEMIC_BUILD=Off \
"

COMPILER_ARGS=''

STATIC=
CLANG_VERSION=11

ARCH=`uname -m`

error() {
    echo "error: $*" >&2
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
        echo "Configuring shared Ubuntu build with Clang-$CLANG_VERSION frontend"
    else
        echo "Configuring static Ubuntu build"
    fi

    if [ $ARCH = "aarch64" ]
    then
	echo "Detected ARM64 Linux!"
	# TODO: We should start using container builds in actions!
	SOLVER_FLAGS="$SOLVER_FLAGS -DENABLE_Z3=On -DZ3_DIR=/usr -DENABLE_GOTO_CONTRACTOR=OFF"
	return
    fi
    
    sudo apt-get update &&
    sudo apt-get install -y $PKGS &&
    echo "Installing Python dependencies" &&
    pip3 install --user meson ast2json &&
    meson --version &&

    BASE_ARGS="$BASE_ARGS \
        -DENABLE_OLD_FRONTEND=Off \
        -DENABLE_PYTHON_FRONTEND=On \
        -DBUILD_STATIC=$STATIC \
    " &&
    SOLVER_FLAGS="$SOLVER_FLAGS \
        -DENABLE_Z3=ON \
    " &&
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

usage() {
    echo "$0 [-OPTS]"
    echo
    echo "Options [defaults]:"
    echo "  -h         display this help message"
    echo "  -b BTYPE   set cmake build type to BTYPE [RelWithDebInfo]"
    echo "  -s STYPE   enable sanitizer STYPE and compile with clang [disabled]"
    echo "  -e ON|OFF  enable/disable -Werror [OFF]"
    echo "  -r ON|OFF  enable/disable 'benchbringup' [OFF]"
    echo "  -d         enable debug output for this script and c2goto"
    echo "  -S ON|OFF  enable/disable static build [ON for Ubuntu, OFF for macOS]"
    echo "  -c VERS    use packaged clang-VERS in a shared build on Ubuntu [11]"
    echo "  -C         build an SV-COMP version [disabled]"
    echo "  -B ON|OFF  enable/disable esbmc bundled libc [ON]"
    echo
    echo "This script prepares the environment, downloads dependencies, configures"
    echo "the ESBMC build and runs the commands to compile and install ESBMC into"
    echo "the directory: $PWD/release"
    echo "Needs to be executed from the top-level directory of ESBMC's source tree."
    echo "The build directory 'build' must not exist and will be created by this script."
    echo
    echo "Supported environments are: Ubuntu-22.04 and macOS."
}

# Setup build flags (release, debug, sanitizer, ...)
while getopts hb:s:e:r:dS:c:CB: flag
do
    case "${flag}" in
    h) usage; exit 0 ;;
    b) BASE_ARGS="$BASE_ARGS -DCMAKE_BUILD_TYPE=${OPTARG}" ;;
    s) BASE_ARGS="$BASE_ARGS -DSANITIZER_TYPE=${OPTARG}"
       COMPILER_ARGS="$COMPILER_ARGS CC=clang CXX=clang++" ;;
    e) BASE_ARGS="$BASE_ARGS -DENABLE_WERROR=${OPTARG}" ;;
    r) BASE_ARGS="$BASE_ARGS -DBENCHBRINGUP=${OPTARG}" ;;
    d) set -x; export ESBMC_OPTS='--verbosity 9' ;;
    S) STATIC=$OPTARG ;; # should be capital ON or OFF
    c) CLANG_VERSION=$OPTARG ;; # LLVM/Clang major version
    C) BASE_ARGS="$BASE_ARGS -DESBMC_SVCOMP=ON"
       SOLVER_FLAGS="\
          -DENABLE_BOOLECTOR=On \
    	  -DENABLE_YICES=ON \
	  -DENABLE_CVC4=OFF \
          -DENABLE_BITWUZLA=On \
	  -DENABLE_Z3=On \
          -DENABLE_Mathsat=ON \
          -DENABLE_GOTO_CONTRACTOR=On \
          -DACADEMIC_BUILD=ON"  ;;
    B) BASE_ARGS="$BASE_ARGS -DESBMC_BUNDLE_LIBC=$OPTARG" ;;
    *) exit 1 ;;
    esac
done
if [ $# -ge $OPTIND ]; then
    shift $((OPTIND-1))
    error "unknown trailing parameters: $*"
fi

# Detect the platform ($OSTYPE was not working on github actions for ubuntu)
# Note: Linux here means Ubuntu, this will mostly not work anywhere else.
OS=`uname`

# Create build directory
mkdir build && cd build || exit $?

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
