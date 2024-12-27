# The ESBMC model checker

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d14d06e975644907a2eb9521e09ccfe4)](https://app.codacy.com/gh/esbmc/esbmc?utm_source=github.com&utm_medium=referral&utm_content=esbmc/esbmc&utm_campaign=Badge_Grade_Dashboard)
[![codecov](https://codecov.io/gh/esbmc/esbmc/branch/master/graph/badge.svg)](https://codecov.io/gh/esbmc/esbmc)

ESBMC (the Efficient SMT-based Context-Bounded Model Checker) is a mature, permissively licensed open-source context-bounded model checker that automatically detects or proves the absence of runtime errors in single- and multi-threaded C, C++, CUDA, CHERI, Kotlin, Python, and Solidity programs. It can automatically verify predefined safety properties (e.g., bounds check, pointer safety, overflow) and user-defined program assertions. 

ESBMC supports: 

- The Clang compiler as its C/C++/CHERI/CUDA frontend;
- The Soot framework via Jimple as its Java/Kotlin frontend;
- The ast module as its [Python frontend](./src/python-frontend/README.md);
- Implements the Solidity grammar production rules as its Solidity frontend;
- Supports IEEE floating-point arithmetic for various SMT solvers.

ESBMC also implements state-of-the-art incremental BMC and *k*-induction proof-rule algorithms based on Satisfiability Modulo Theories (SMT) and Constraint Programming (CP) solvers.

We provide some background material/publications to help you understand exactly what ESBMC can offer. These are available [online](https://ssvlab.github.io/esbmc/publications.html). You can also check the ESBMC [architecture](https://github.com/esbmc/esbmc/blob/master/ARCHITECTURE.md) for further information about our main components.

Our main website is [esbmc.org](http://esbmc.org). 

### How to build/install ESBMC

#### Ubuntu 24.04

To compile ESBMC on Ubuntu 24.04 with LLVM 14 and the SMT solver Z3:

````
sudo apt update
sudo apt-get install -y clang-14 llvm-14 clang-tidy-14 python-is-python3 python3 git ccache unzip wget curl bison flex g++-multilib linux-libc-dev libboost-all-dev libz3-dev libclang-14-dev libclang-cpp-dev cmake
git clone https://github.com/esbmc/esbmc.git
mkdir build && cd build
cmake .. -DENABLE_Z3=1
make -j4
````

#### Fedora 40

To compile ESBMC on Fedora 40 with the latest version of LLVM and the SMT solver Z3:

```sh
# Warning, the --allowerasing parameter will also remove incompatible packages to the packages specified below
sudo dnf install --best --allowerasing "@Development Tools" clang llvm llvm-devel clang-tools-extra python3 git ccache unzip wget curl bison flex gcc-c++ glibc-devel glibc-devel.i686 boost-devel boost-devel.i686 z3-devel clang-devel clang-devel.i686 cmake zlib-devel libffi-devel libstdc++-devel libstdc++-devel.i686

cmake .. -DENABLE_Z3=1 -DZ3_DIR=/usr/include/z3

make -j4
```

To build ESBMC with other operating systems and SMT solvers, please see the [BUILDING](https://github.com/esbmc/esbmc/blob/master/BUILDING.md) file. 

The user can also download the latest ESBMC binary for Ubuntu and Windows from the [releases page](https://github.com/esbmc/esbmc/releases).

#### FreeBSD

ESBMC should compile just fine in FreeBSD as long as the 32-bit libraries are enabled

```sh
pkg install git cmake python3 z3 bison flex boost-all
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.0/clang+llvm-16.0.0-amd64-unknown-freebsd13.tar.xz && mv clang16
mkdir build && cd build
cmake .. -DLLVM_DIR=../clang16 -DClang_DIR=../clang16
make -j4
```

#### Mac OS X

ESBMC works to some extent on ARM64 (M1/M2/M3/M4) Macs, assuming you have installed the MAC OS Dev tools. However, the compile option for GOTO_SYSROOT needs to be changed. Note that make -j8 can be increased to -j32 on faster Macs.
````
brew install z3
brew install bison
brew install clang
brew install llvm

git clone https://github.com/esbmc/esbmc.git
mkdir build && cd build

````

##### Execution for Mac with Apple chips
````sh
cmake .. -DENABLE_Z3=1 -DC2GOTO_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm -DClang_DIR=/opt/homebrew/opt/llvm/lib/cmake/clang

make -j8
````

##### Execution for Mac with Intel chips
````sh

# replace path_llvm with the path of llvm, e.g. "/usr/local/opt/llvm"  you can get it with this command "brew --prefix llvm"

cmake .. -DCMAKE_PREFIX_PATH="{path_llvm}" -DENABLE_Z3=1 -DC2GOTO_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
make -j8
sudo make install
esbmc --version
````

We recommend using AMD64 via docker for optimal performance. Sample docker-compose and docker files follow:

DockerFile sample:

````
FROM node:18-slim

# Install dependencies for ESBMC and other build tools
RUN apt-get update && apt-get install -y \
    clang-14 \
    llvm-14 \
    clang-tidy-14 \
    python-is-python3 \
    python3 \
    git \
    ccache \
    unzip \
    wget \
    curl \
    bison \
    flex \
    g++-multilib \
    linux-libc-dev \
    libboost-all-dev \
    libz3-dev \
    libclang-14-dev \
    libclang-cpp-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Keep container running with tail -f /dev/null
CMD ["bash", "-c", "tail -f /dev/null"]
````

Docker compose file:
````
version: '3.8'
services:
  esbmc:
    platform: linux/amd64
    build:
      context: .
      dockerfile: Dockerfile  # Assuming your Dockerfile is named `Dockerfile`
    tty: true
    stdin_open: true
````
The linux/amd64 line is very important, to virtualize amd64. Now do docker-compose up --build. You can then follow the linux instructions. Make -j16 works well on M2 mac's and beyond.

### How to use ESBMC

As an illustrative example to show some of the ESBMC features, consider the following C code:

````C
#include <assert.h>
#include <math.h>
int main() {
  unsigned int N = nondet_uint();
  double x = nondet_double();
  if(x <= 0 || isnan(x))
    return 0;
  unsigned int i = 0;
  while(i < N) {
    x = (2*x);
    assert(x>0);
    ++i;
  }
  assert(x>0);
  return 0;
}
````

Here, ESBMC is invoked as follows:

````
$esbmc file.c --k-induction
````

Where `file.c` is the C program to be checked, and --k-induction selects the k-induction proof rule. The user can choose the SMT solver, property, and verification strategy. Note that you need math.h installed on your system, especially if you run a release version. build-essential typically covers math.h.

For this particular C program, ESBMC provides the following output as the verification result:

````
*** Checking inductive step
Starting Bounded Model Checking
Unwinding loop 2 iteration 1 file ex5.c line 8 function main
Not unwinding loop 2 iteration 2 file ex5.c line 8 function main
Symex completed in: 0.001s (40 assignments)
Slicing time: 0.000s (removed 16 assignments)
Generated 2 VCC(s), 2 remaining after simplification (24 assignments)
No solver specified; defaulting to Boolector
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.005s
Solving with solver Boolector 3.2.0
Encoding to solver time: 0.005s
Runtime decision procedure: 0.427s
BMC program time: 0.435s

VERIFICATION SUCCESSFUL

Solution found by the inductive step (k = 2)
````

We refer the user to our [documentation webpage](https://ssvlab.github.io/esbmc/documentation.html) for further examples of the ESBMC's features.

### Using Config Files

ESBMC supports specifying options through TOML formatted config files. To use a config file, export an environment variable:

```sh
export ESBMC_CONFIG_FILE="path/to/config.toml"
```

If no environment file is specified, then the default locations will be checked:

* Windows: `%userprofile%\esbmc.toml`
* UNIX: `~/.config/esbmc.toml`

If nothing is found, then nothing is loaded. If you set the environment variable to
the empty string, then it disables the entire config file loading process.

```sh
export ESBMC_CONFIG_FILE=""
```

An example of a config file:

```toml
interval-analysis = true
goto-unwind = true
unlimited-goto-unwind = true
k-induction = true
state-hashing = true
add-symex-value-sets = true
k-step = 2
floatbv = true
unlimited-k-steps = false
max-k-step = 100
memory-leak-check = true
context-bound = 2
```

### Features

ESBMC detects errors in software by simulating a finite prefix of the program execution with all possible inputs. Classes of implementation errors that can be detected include:
 * User-specified assertion failures
 * Out-of-bounds array access
 * Illegal pointer dereferences, such as:
   * Dereferencing null
   * Performing an out-of-bounds dereference
   * Double-free of malloc'd memory
   * Misaligned memory access
 * Integer overflows
 * Undefined behavior on shift operations
 * Floating-point for NaN
 * Divide by zero
 * Memory leaks

Concurrent software (using the pthread API) is verified by explicitly exploring interleavings, producing one symbolic execution per interleaving. By default, pointer-safety, array-out-of-bounds, division-by-zero, and user-specified assertions will be checked for; one can also specify options to check concurrent programs for:
 * Deadlock (only on pthread mutexes and convars)
 * Data races (i.e., competing writes)
 * Atomicity violations at visible assignments
 * Lock acquisition ordering

By default, ESBMC performs a "lazy" depth-first search of interleavings -- it can also encode (explicitly) all interleavings into a single SMT formula.

Many SMT solvers are currently supported:
 * Z3 4.13+
 * Bitwuzla
 * Boolector 3.0+
 * MathSAT
 * CVC4
 * CVC5
 * Yices 2.2+

In addition, ESBMC can be configured to use the SMTLIB interactive text format with a pipe to communicate with an arbitrary solver process, although there are not insignificant overheads involved.

### Tutorials

We provide a short video that explains ESBMC: 

https://www.youtube.com/watch?v=uJ5Jn0sxm08&t=2182s

In a workshop between Arm Research and the University of Manchester, this video was delivered as part of a technical talk on exploiting the SAT revolution for automated software verification.

We offer a post-graduate course in software security that explains the internals of ESBMC. 

https://ssvlab.github.io/lucasccordeiro/courses/2020/01/software-security/index.html

This course unit introduces students to basic and advanced approaches to formally building verified trustworthy software systems, where trustworthiness comprises five attributes: *reliability*, *availability*, *safety*, *resilience* and *security*.

### Selected Publications

* Yiannis Charalambous, Norbert Tihanyi, Ridhi Jain, Youcheng Sun, Mohamed Amine Ferrag, Lucas C. Cordeiro. [A New Era in Software Security: Towards Self-Healing Software via Large Language Models and Formal Verification](https://arxiv.org/pdf/2305.14752.pdf). Technical Report, CoRR abs/2305.14752, 2023. [DOI](https://doi.org/10.48550/arXiv.2305.14752)
  
* Rafael Menezes, Daniel Moura, Helena Cavalcante, Rosiane de Freitas, Lucas C. Cordeiro . [ESBMC-Jimple: verifying Kotlin programs via jimple intermediate representation](https://dl.acm.org/doi/abs/10.1145/3533767.3543294) In ISSTA'22, pp. 777-780, 2022. [DOI](https://doi.org/10.1145/3533767.3543294)

* Franz Brauße, Fedor Shmarov, Rafael Menezes, Mikhail R. Gadelha, Konstantin Korovin, Giles Reger, Lucas C. Cordeiro. [ESBMC-CHERI: towards verification of C programs for CHERI platforms with ESBMC](https://dl.acm.org/doi/abs/10.1145/3533767.3543289) In ISSTA'22, pp. 773-776, 2022. [DOI](https://doi.org/10.1145/3533767.3543289) 

* Felipe R. Monteiro, Mikhail R. Gadelha, Lucas C. Cordeiro. [Model checking C++ programs.](https://onlinelibrary.wiley.com/doi/epdf/10.1002/stvr.1793) In Softw. Test. Verification Reliab. 32(1), 2022. [DOI](https://doi.org/10.1002/stvr.1793), [Video](https://www.youtube.com/watch?v=cX31c976tjM), **Open access**.

* Mikhail R. Gadelha, Lucas C. Cordeiro, Denis A. Nicole. [An Efficient Floating-Point Bit-Blasting API for Verifying C Programs.](https://ssvlab.github.io/lucasccordeiro/papers/nsv2020.pdf) In VSTTE, pp. 178-195, 2020. [DOI](https://doi.org/10.1007/978-3-030-63618-0_11)

* Mikhail Y. R. Gadelha, Felipe R. Monteiro, Jeremy Morse, Lucas C. Cordeiro, Bernd Fischer, Denis A. Nicole. [ESBMC 5.0: an industrial-strength C model checker.](https://ssvlab.github.io/lucasccordeiro/papers/ase2018.pdf) In ASE, pp. 888-891, 2018. [DOI](https://doi.org/10.1145/3238147.3240481) 

* Jeremy Morse, Lucas C. Cordeiro, Denis A. Nicole, Bernd Fischer. [Model checking LTL properties over ANSI-C programs with bounded traces.](https://ssvlab.github.io/lucasccordeiro/papers/sosym2013.pdf) In Softw. Syst. Model. 14(1), pp. 65-81, 2015. [DOI](https://doi.org/10.1007/s10270-013-0366-0)

* Mikhail Y. R. Gadelha, Hussama Ibrahim Ismail, Lucas C. Cordeiro. [Handling loops in bounded model checking of C programs via k-induction.](https://ssvlab.github.io/lucasccordeiro/papers/sttt2017.pdf) In Int. J. Softw. Tools Technol. Transf. 19(1), pp. 97-114, 2017. [DOI](https://doi.org/10.1007/s10009-015-0407-9)

* Phillipe A. Pereira, Higo F. Albuquerque, Isabela da Silva, Hendrio Marques, Felipe R. Monteiro, Ricardo Ferreira, Lucas C. Cordeiro. [SMT-based context-bounded model checking for CUDA programs.](https://ssvlab.github.io/lucasccordeiro/papers/cppe2017.pdf) In Concurr. Comput. Pract. Exp. 29(22), 2017. [DOI](https://doi.org/10.1002/cpe.3934)

* Lucas C. Cordeiro, Bernd Fischer, João Marques-Silva. [SMT-Based Bounded Model Checking for Embedded ANSI-C Software.](https://ssvlab.github.io/lucasccordeiro/papers/tse2012.pdf) In IEEE Trans. Software Eng. 38(4), pp. 957-974, 2012. [DOI](https://doi.org/10.1109/TSE.2011.59)

* Lucas C. Cordeiro, Bernd Fischer. [Verifying multi-threaded software using smt-based context-bounded model checking.](https://ssvlab.github.io/lucasccordeiro/papers/icse2011.pdf) In ICSE, pp. 331-340, 2011. [DOI](https://doi.org/10.1145/1985793.1985839)

### Awards

* 35 awards from international competitions on software verification (SV-COMP) and testing (Test-Comp) 2012-2024 at TACAS/FASE (Strength: Bug Finding and Code Coverage).
* Most Influential Paper Award at ASE’23
* Best Tool Paper Award at SBSeg'23
* Best Paper Award at SBESC’15
* Distinguished Paper Award at ICSE’11
   
### ESBMC-CHERI Video & Download

This [video](https://youtu.be/CsWHnmU4UMs) describes how to obtain, build and run ESBMC-CHERI on an example.

A pre-compiled binary for Linux is available in the pre-release
[ESBMC-CHERI](https://github.com/esbmc/esbmc/releases/tag/v6.9-cheri), for other
systems/archs the [BUILDING.md](https://github.com/esbmc/esbmc/blob/cheri-clang/BUILDING.md)
document explains the necessary installation steps.

# Open source

ESBMC is open-source software mainly distributed under the Apache License 2.0. It contains a significant amount of other people's software. However, please take a look at the COPYING file to explain who owns what and under what terms it is distributed.

We'd be extremely happy to receive contributions to improve ESBMC (under the terms of the Apache License 2.0). If you'd like to submit anything, please file a pull request against the public GitHub repo. General discussion and release announcements will be made via GitHub. Please post an issue on GitHub and contact us about research or collaboration.

Please review the [developer documentation](https://github.com/esbmc/esbmc/blob/master/CONTRIBUTIONS.md) if you want to contribute to ESBMC.

### Differences from CBMC

ESBMC is a fork of CBMC v2.9 (2008), the C Bounded Model Checker. The primary differences between the two are described [here](https://github.com/esbmc/esbmc/blob/master/CBMC.md).

### Acknowledgments

ESBMC is a joint project of the Federal University of Amazonas (Brazil), the University of Manchester (UK), the University of Southampton (UK), and the University of Stellenbosch (South Africa).

The ESBMC development was supported by various research funding agencies, including CNPq (Brazil), CAPES (Brazil), FAPEAM (Brazil), EPSRC (UK), Royal Society (UK), British Council (UK), European Commission (Horizon 2020), and companies including ARM, Intel, Motorola Mobility, Nokia Institute of Technology and Samsung. The ESBMC development is currently funded by ARM, EPSRC grants [EP/T026995/1](https://enncore.github.io) and [EP/V000497/1](https://scorch-project.github.io), [Ethereum Foundation](https://blog.ethereum.org/2022/07/29/academic-grants-grantee-announce), [EU H2020 ELEGANT 957286](https://www.elegant-h2020.eu), Intel, Motorola Mobility (through Agreement N° 4/2021), [Soteria project](https://soteriaresearch.org) awarded by the UK Research and Innovation for the Digital Security by Design (DSbD) Programme, and XC5 Hong Kong Limited.
