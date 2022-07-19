FROM ubuntu:20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y clang-tidy python-is-python3 csmith python3 git ccache unzip wget curl libcsmith-dev gperf libgmp-dev cmake bison flex gcc-multilib linux-libc-dev libboost-all-dev ninja-build python3-setuptools libtinfo-dev pkg-config python3-pip && pip install toml
WORKDIR /workspace
# Boolector 3.2.2
RUN git clone --depth=1 --branch=3.2.2 https://github.com/boolector/boolector && cd boolector && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./configure.sh --prefix $PWD/../boolector-release && cd build && make -j4 && make install
# Z3 4.8.9
RUN wget https://github.com/Z3Prover/z3/releases/download/z3-4.8.9/z3-4.8.9-x64-ubuntu-16.04.zip && unzip z3-4.8.9-x64-ubuntu-16.04.zip && mv z3-4.8.9-x64-ubuntu-16.04/ z3
# Yices 2.6.4
RUN wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz && tar xf gmp-6.1.2.tar.xz && rm gmp-6.1.2.tar.xz && cd gmp-6.1.2 && ./configure --prefix $PWD/../gmp --disable-shared ABI=64 CFLAGS=-fPIC CPPFLAGS=-DPIC && make -j4 && make install
RUN git clone https://github.com/SRI-CSL/yices2.git && cd yices2 && git checkout Yices-2.6.4 && autoreconf && ./configure --prefix /workspace/yices  --with-static-gmp=$PWD/../gmp/lib/libgmp.a && make -j4 && make static-lib && make install && cp ./build/x86_64-pc-linux-gnu-release/static_lib/libyices.a ../yices/lib
# CVC4 (b826fc8ae95fc) 
RUN apt-get install -y openjdk-11-jdk
RUN git clone https://github.com/CVC4/CVC4.git && cd CVC4 && git reset --hard b826fc8ae95fc && ./contrib/get-antlr-3.4 && ./configure.sh --optimized --prefix=../cvc4 --static --no-static-binary && cd build && make -j4 && make install
# Bitwuzla (smtcomp-2021)
RUN git clone --depth=1 --branch=smtcomp-2021 https://github.com/bitwuzla/bitwuzla.git && cd bitwuzla && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./contrib/setup-symfpu.sh && ./configure.sh --prefix $PWD/../bitwuzla-release && cd build && cmake -DGMP_INCLUDE_DIR=$PWD/../../gmp/include -DGMP_LIBRARIES=$PWD/../../gmp/lib/libgmp.a -DONLY_LINGELING=ON ../ && make -j8 && make install
# Download clang11
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz  && tar xf clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz && mv clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04 clang && rm clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
# Python is Python3
RUN apt-get install -y python-is-python3
# Setup ibex 2.8.9
RUN wget http://www.ibex-lib.org/ibex-2.8.9.tgz && tar xvfz ibex-2.8.9.tgz && cd ibex-2.8.9 && ./waf configure --lp-lib=soplex && ./waf install  
# Clang-tidy
