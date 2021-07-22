sudo apt-get update && sudo apt-get install -y gperf libgmp-dev cmake bison flex linux-libc-dev libboost-all-dev ninja-build python3-setuptools libtinfo-dev
git clone --depth=1 --branch=3.2.1 https://github.com/boolector/boolector && cd boolector && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./configure.sh --prefix $PWD/../boolector-release && cd build && make -j4 && make install
cd ../../
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-aarch64-linux-gnu.tar.xz
tar xf clang+llvm-11.0.0-aarch64-linux-gnu.tar.xz && mv clang+llvm-11.0.0-aarch64-linux-gnu clang
mkdir build && cd build
cmake .. -DClang_DIR=$PWD/../clang -DLLVM_DIR=$PWD/../clang -DBUILD_STATIC=On -DBoolector_DIR=$PWD/../boolector-release -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../release -DCMAKE_BUILD_TYPE=Debug
make -j2 . && make install
