#!/bin/bash

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed!"
    echo "Please install Homebrew first by running this command:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo "After installing Homebrew, run this script again."
    exit 1
fi

# Ask about Boolector right at the start (Y/yes is default)
read -p "Do you want to install the recommended Boolector solver? [Y/n]: " use_boolector
use_boolector=${use_boolector:-Y}  # Default to Y if user just hits enter

# Create and enter build directory
echo "Creating build directory..."
mkdir -p build
cd build

echo "Installing ESBMC dependencies..."
brew install z3 bison clang llvm

# Get number of CPUs and add 1
CPU_COUNT=$(($(sysctl -n hw.ncpu) + 1))

# Get paths
PATH_LLVM=$(brew --prefix llvm)
PATH_SDK=$(xcrun --show-sdk-path)
PATH_Z3=$(brew --prefix z3)

# Function to install Boolector
install_boolector() {
    echo "Installing Boolector..."
    cd ..
    git clone --depth=1 --branch=3.2.3 https://github.com/boolector/boolector
    cd boolector
    ./contrib/setup-lingeling.sh
    ./contrib/setup-btor2tools.sh
    ./configure.sh --prefix $PWD/../boolector-release
    cd build
    make -j${CPU_COUNT}
    make install
    cd ../../build
}

if [[ $use_boolector =~ ^[Yy]$ ]]; then
    install_boolector
    echo "Building ESBMC with Boolector..."
    cmake .. \
        -DCMAKE_PREFIX_PATH="$PATH_LLVM" \
        -DENABLE_Z3=1 \
        -DZ3_DIR="$PATH_Z3" \
        -DENABLE_BOOLECTOR=1 \
        -DBoolector_DIR="$PWD/../boolector-release" \
        -DC2GOTO_SYSROOT="$PATH_SDK" \
        -DClang_DIR="$PATH_LLVM/lib/cmake/clang"
else
    echo "Building ESBMC with Z3 only..."
    cmake .. \
        -DCMAKE_PREFIX_PATH="$PATH_LLVM" \
        -DENABLE_Z3=1 \
        -DZ3_DIR="$PATH_Z3" \
        -DC2GOTO_SYSROOT="$PATH_SDK" \
        -DClang_DIR="$PATH_LLVM/lib/cmake/clang"
fi

echo "Running make..."
make -j${CPU_COUNT}

echo "Installing ESBMC system-wide (requires sudo permission)..."
sudo make install

echo "Build and installation complete! You can now run 'esbmc' from anywhere."
