###################################
# NOTES
###################################

# This docker environment will be used as a base to build ESBMC in static mode.
# It currently works by providing an ubuntu 18.04 environment with the following tools:

# - Clang 9
# - Boolector

###################################
# START
###################################
FROM jenkins/jnlp-slave

###################################
# ROOT
###################################
USER root
# Core Packages
RUN apt-get update && apt-get install --no-install-recommends  -y  \
    git \
    build-essential \
    wget \
    bison \
    flex \
    unzip \
    gcc-multilib \
    linux-libc-dev \
    libzip-dev \
    libtinfo-dev \
    libxml2-dev \
    libboost-all-dev \
    automake \
    pkg-config \
    libtool \
# CMake Packages
    cmake ninja-build \
    gperf libgmp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 


# Download Clang 9
WORKDIR /usr/local/opt
RUN  wget http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && tar xf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && mv clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04 clang9 && rm clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
ENV CLANG_HOME=/usr/local/opt/clang9

###################################
# jenkins user
###################################
USER jenkins

##################################
# Building Boolector
##################################
WORKDIR /home/jenkins
RUN git clone https://github.com/boolector/boolector && cd boolector && git reset --hard 3.2.0 && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./configure.sh --prefix $HOME/boolector-3.2.0 && cd build && make -s -j4 && make install

##################################
# ENV Vars
##################################
ENV BTOR_DIR=/home/jenkins/boolector

# Execute jenkins-slave
ENTRYPOINT ["/usr/local/bin/jenkins-slave"]