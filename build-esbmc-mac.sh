#!/bin/bash
cd ..
# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed!"
    echo "Please install Homebrew first by running this command:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo "After installing Homebrew, run this script again."
    exit 1
fi

# Ask about Python frontend and Boolector right at the start (Y/yes is default for both)
read -p "Do you want to build the Python frontend? [Y/n]: " use_python
use_python=${use_python:-Y}  # Default to Y if user just hits enter


read -p "Do you want to install the recommended Boolector solver? [Y/n]: " use_boolector
use_boolector=${use_boolector:-Y}  # Default to Y if user just hits enter

# Install CMake
brew install cmake

# Create and enter build directory
echo "Creating build directory..."
mkdir -p build
cd build

echo "Installing ESBMC dependencies..."
brew install z3 bison clang llvm

# Set up Python environment if requested
if [[ $use_python =~ ^[Yy]$ ]]; then
    echo "Setting up Python environment..."
    python3 -m venv ../esbmc-venv
    source ../esbmc-venv/bin/activate
    pip install ast2json
    deactivate
fi

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

# Build configuration
CMAKE_ARGS=(
    -DCMAKE_PREFIX_PATH="$PATH_LLVM"
    -DENABLE_Z3=1
    -DZ3_DIR="$PATH_Z3"
    -DC2GOTO_SYSROOT="$PATH_SDK"
    -DClang_DIR="$PATH_LLVM/lib/cmake/clang"
)

# Add Python frontend if requested
if [[ $use_python =~ ^[Yy]$ ]]; then
    CMAKE_ARGS+=(-DENABLE_PYTHON_FRONTEND=On)
fi

# Add Boolector if requested
if [[ $use_boolector =~ ^[Yy]$ ]]; then
    install_boolector
    echo "Building ESBMC with Boolector..."
    CMAKE_ARGS+=(
        -DENABLE_BOOLECTOR=1
        -DBoolector_DIR="$PWD/../boolector-release"
    )
else
    echo "Building ESBMC with Z3 only..."
fi

# Run cmake with all arguments
cmake .. "${CMAKE_ARGS[@]}"


echo "Running make..."
make -j${CPU_COUNT}

echo "Installing ESBMC system-wide (requires sudo permission)..."
sudo make install

echo "Build and installation complete! You can now run 'esbmc' from anywhere."

# Print additional instructions for Python frontend if it was installed
if [[ $use_python =~ ^[Yy]$ ]]; then
    echo -e "\nTo use the Python frontend, you need to activate the virtual environment:"
    echo "source esbmc-venv/bin/activate"
    echo "Then you can use ESBMC with Python files"
fi
