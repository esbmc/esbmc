---
title: Build Guide
weight: 1
---

ESBMC builds on Linux, macOS, FreeBSD and Windows. Pick your platform below and
follow the steps. Each tab produces a working `esbmc` with the Z3 solver; the
heavier dependencies (LLVM/Clang, fmt, nlohmann-json, yaml-cpp and immer) are
downloaded and built automatically with `-DDOWNLOAD_DEPENDENCIES=1` where the
platform supports it.

> [!NOTE]
> At least 6 GB of RAM is recommended. ESBMC is distributed mainly under the
> [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

For a build with several solvers, or CHERI/fuzzing support, see
[Building with all solvers](#building-with-all-solvers) and [Advanced](#advanced).

## Build ESBMC

{{< tabs >}}

{{< tab name="Ubuntu / Debian" >}}
{{% steps %}}

### Install prerequisites

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git bison flex \
  python3 libboost-all-dev g++-multilib
```

LLVM/Clang, Z3 and the libraries fmt, nlohmann-json, yaml-cpp and immer are
fetched by `-DDOWNLOAD_DEPENDENCIES=1`, so they are not installed here.

### Get the source

```sh
git clone https://github.com/esbmc/esbmc.git
cd esbmc
```

### Configure and build

```sh
cmake -GNinja -Bbuild -DDOWNLOAD_DEPENDENCIES=1 -DENABLE_Z3=1
ninja -C build
```

The binary is written to `build/src/esbmc/esbmc`.

{{% details title="Optional: run the regression tests" closed="true" %}}
Add the testing flags when configuring, then run ctest:

```sh
cmake -GNinja -Bbuild -DDOWNLOAD_DEPENDENCIES=1 -DENABLE_Z3=1 \
  -DBUILD_TESTING=On -DENABLE_REGRESSION=On
ninja -C build
ctest --test-dir build -j"$(nproc)" -L esbmc --timeout 120
```
{{% /details %}}

{{% details title="Optional: enable a frontend or extra solvers" closed="true" %}}
For Python, Solidity or IBEX support see [Optional frontends](#optional-frontends).
For a multi-solver build see [Building with all solvers](#building-with-all-solvers).
{{% /details %}}

{{% /steps %}}
{{< /tab >}}

{{< tab name="Fedora" >}}
{{% steps %}}

### Install prerequisites

```sh
sudo dnf install gcc-c++ cmake ninja-build git bison flex python3 boost-devel \
  z3-devel glibc-devel.i686 libstdc++-devel.i686
```

LLVM/Clang, fmt, nlohmann-json, yaml-cpp and immer are fetched by
`-DDOWNLOAD_DEPENDENCIES=1`; Z3 comes from the system `z3-devel` package. The
flag is required on Fedora because `immer` has no Fedora package.
`glibc-devel.i686` and `libstdc++-devel.i686` are the 32-bit C runtime headers
used to build ESBMC's 32-bit operational models.

### Get the source

```sh
git clone https://github.com/esbmc/esbmc.git
cd esbmc
```

### Configure and build

```sh
cmake -GNinja -Bbuild -DDOWNLOAD_DEPENDENCIES=1 -DENABLE_Z3=1 -DZ3_DIR=/usr/include/z3
ninja -C build
```

The binary is written to `build/src/esbmc/esbmc`.

{{% details title="Optional: skip 32-bit support" closed="true" %}}
If you do not need 32-bit verification, drop the two `.i686` packages and add
`-DENABLE_BUNDLE_LIBC_32BIT=Off`.
{{% /details %}}

{{% details title="Optional: enable a frontend or extra solvers" closed="true" %}}
For Python, Solidity or IBEX support see [Optional frontends](#optional-frontends).
For a multi-solver build see [Building with all solvers](#building-with-all-solvers).
{{% /details %}}

{{% /steps %}}
{{< /tab >}}

{{< tab name="macOS" >}}
{{% steps %}}

### Install prerequisites

```sh
brew install llvm@21 z3 boost cmake ninja python
```

> [!NOTE]
> On macOS, `-DDOWNLOAD_DEPENDENCIES` must be combined with `-DLLVM_DIR` so only
> the portable libraries are downloaded — the prebuilt LLVM/Z3 it would
> otherwise fetch are Linux binaries.

### Get the source

```sh
git clone https://github.com/esbmc/esbmc.git
cd esbmc
```

### Configure and build

```sh
cmake -GNinja -Bbuild -DDOWNLOAD_DEPENDENCIES=1 -DENABLE_Z3=1 \
  -DLLVM_DIR=$(brew --prefix llvm@21)/lib/cmake/llvm \
  -DClang_DIR=$(brew --prefix llvm@21)/lib/cmake/clang \
  -DZ3_DIR=$(brew --prefix z3) \
  -DC2GOTO_SYSROOT=$(xcrun --show-sdk-path)
ninja -C build
```

{{% details title="Easiest: use the helper script" closed="true" %}}
The repository ships `scripts/build-esbmc-mac.sh`, which creates the build
folder, optionally installs Boolector/Bitwuzla, and installs `esbmc` globally:

```sh
./scripts/build-esbmc-mac.sh
```
{{% /details %}}

{{% details title="Xcode SDK" closed="true" %}}
A full Xcode or the Command Line Tools SDK is required. If
`xcrun --show-sdk-path` fails, install the tools with `xcode-select --install`.
{{% /details %}}

{{% /steps %}}
{{< /tab >}}

{{< tab name="FreeBSD" >}}
{{% steps %}}

### Install prerequisites

```sh
pkg install git cmake ninja python3 z3 bison flex boost-all
```

ESBMC needs 32-bit libraries on FreeBSD — make sure the `lib32` set is installed.

### Provide an LLVM/Clang toolchain

ESBMC's prebuilt LLVM download targets Linux, so fetch a FreeBSD LLVM release and
point the build at it:

```sh
fetch https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.0/clang+llvm-16.0.0-amd64-unknown-freebsd13.tar.xz
tar xf clang+llvm-16.0.0-amd64-unknown-freebsd13.tar.xz
mv clang+llvm-16.0.0-amd64-unknown-freebsd13 clang16
```

### Get the source, configure and build

```sh
git clone https://github.com/esbmc/esbmc.git && cd esbmc
cmake -GNinja -Bbuild -DDOWNLOAD_DEPENDENCIES=1 -DENABLE_Z3=1 -DZ3_DIR=/usr/local \
  -DLLVM_DIR=$PWD/../clang16 -DClang_DIR=$PWD/../clang16
ninja -C build
```

{{% /steps %}}
{{< /tab >}}

{{< tab name="Windows" >}}
{{% steps %}}

### Install the toolchain

- [Visual Studio](https://visualstudio.microsoft.com/) with the **Desktop
  development with C++** workload (install the English language pack even on
  non-English systems).
- [Chocolatey](https://chocolatey.org/) and
  [vcpkg](https://github.com/microsoft/vcpkg) (integrate vcpkg with your system).
- Check the repository out with **LF** line endings — CRLF will not work.

### Install dependencies

```bat
vcpkg.exe install boost-filesystem:x64-windows boost-date-time:x64-windows boost-test:x64-windows boost-multi-index:x64-windows boost-crc:x64-windows boost-property-tree:x64-windows boost-uuid:x64-windows
choco install winflexbison gnuwin32-coreutils.portable
```

Also obtain [Z3](https://github.com/Z3Prover/z3/releases) and a prebuilt
LLVM/Clang. The official LLVM releases do not ship the CMake modules ESBMC
needs, so build LLVM from source (see below).

### Configure and build

Open the CMake project in Visual Studio, set the configuration to **Release**,
the toolchain to **msvc_x64_x64**, the triplet to **x64-windows**, and add the
dependency paths to the CMake flags:

```
-DLLVM_DIR=C:\Deps\llvm -DClang_DIR=C:\Deps\llvm -DZ3_DIR=C:\Deps\z3
```

{{% details title="Build LLVM/Clang from source on Windows" closed="true" %}}
Download the `llvm-14.X.X.src` and `clang-14.X.X.src` archives from the
[LLVM releases](https://releases.llvm.org/) and unpack each into its own
space-free directory. Then, from the **x64 Native Tools Command Prompt for
VS 2019**:

```bat
mkdir llvm-14.0.6.src\build-release && cd llvm-14.0.6.src\build-release
cmake .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 ^
  -DCMAKE_INSTALL_PREFIX=C:\Deps\llvm -DCMAKE_PREFIX_PATH=C:\Deps\llvm ^
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_LIBXML2=OFF ^
  -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_CRT_RELEASE=MT
msbuild /m -p:Configuration=Release INSTALL.vcxproj
```

Repeat for the `clang-14.X.X.src` tree (same flags) to install Clang into the
same prefix.
{{% /details %}}

{{% details title="Build inside WSL" closed="true" %}}
With [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) you can
build ESBMC exactly as on Linux — follow the **Ubuntu / Debian** tab.
{{% /details %}}

{{% /steps %}}
{{< /tab >}}

{{< /tabs >}}

## Dependency reference

| package   | required | minimum version |
|-----------|----------|-----------------|
| clang     | yes      | 11.0.0          |
| boost     | yes      | 1.77            |
| CMake     | yes      | 3.18.0          |
| Boolector | no       | 3.2.2           |
| CVC4      | no       | 1.8             |
| CVC5      | no       | 1.1.2           |
| MathSAT   | no       | 5.5.4           |
| Yices     | no       | 2.6.4           |
| Z3        | no       | 4.13.3          |
| Bitwuzla  | no       | 0.9.0           |

The version requirements are stable but can change between releases. For all
available CMake options, see
[Options.cmake](https://github.com/esbmc/esbmc/blob/master/scripts/cmake/Options.cmake).

## Optional frontends

These are extra CMake flags added to the configure step in any platform tab above.

{{% details title="Python" closed="true" %}}
Add `-DENABLE_PYTHON_FRONTEND=On` to the configure step:

```sh
cmake -GNinja -Bbuild -DDOWNLOAD_DEPENDENCIES=1 -DENABLE_Z3=1 -DENABLE_PYTHON_FRONTEND=On
```

`ast2json` is vendored in the source tree, so no `pip install` is needed.
Optionally install [mypy](https://mypy-lang.org/) (e.g. `pipx install mypy`) for
Python type checks.
{{% /details %}}

{{% details title="Solidity" closed="true" %}}
Add `-DENABLE_SOLIDITY_FRONTEND=On` to the configure step. The Solidity frontend
verifies smart contracts against predefined safety properties (bounds, overflow,
underflow) and user-defined assertions.
{{% /details %}}

{{% details title="IBEX (interval constraint solving)" closed="true" %}}
Enables `--goto-contractor`. Install IBEX following its
[instructions](http://ibex-team.github.io/ibex-lib/install.html), then:

```sh
cmake -GNinja -Bbuild -DENABLE_GOTO_CONTRACTOR=ON -DIBEX_DIR=path-to-ibex
```
{{% /details %}}

## Building with all solvers

ESBMC supports **Bitwuzla**, **Boolector**, **CVC4**, **CVC5**, **MathSAT**,
**Yices 2** and **Z3**. All are optional, but without at least one solver ESBMC
cannot verify most programs. For a single-solver build, the platform tabs above
are enough.

The recipe below mirrors the multi-solver build used in ESBMC's CI: build each
solver into the project directory, then point the configure step at them. Build
only the solvers you need.

{{% steps %}}

### Prepare a Clang/LLVM toolchain

For a static build, let ESBMC download Clang/LLVM:

```sh
ESBMC_CLANG=-DDOWNLOAD_DEPENDENCIES=On
ESBMC_STATIC=ON
```

For a shared build, use the system LLVM/Clang instead (Ubuntu example):

```sh
sudo apt-get install libclang-cpp16-dev
ESBMC_CLANG="-DLLVM_DIR=/usr/lib/llvm-16/lib/cmake/llvm -DClang_DIR=/usr/lib/cmake/clang-16"
ESBMC_STATIC=OFF
```

### Build the solvers

{{% details title="Boolector" closed="true" %}}
```sh
git clone --depth=1 --branch=3.2.3 https://github.com/boolector/boolector && cd boolector && ./contrib/setup-lingeling.sh && ./contrib/setup-btor2tools.sh && ./configure.sh --prefix $PWD/../boolector-release && cd build && make -j9 && make install && cd ../..
```
{{% /details %}}

{{% details title="Z3" closed="true" %}}
```sh
# Linux
wget https://github.com/Z3Prover/z3/releases/download/z3-4.13.3/z3-4.13.3-x64-glibc-2.35.zip && unzip z3-4.13.3-x64-glibc-2.35.zip && mv z3-4.13.3-x64-glibc-2.35 z3
# macOS
brew install z3 && cp -rp $(brew info z3 | egrep "/usr[/a-zA-Z\.0-9]+ " -o) z3
```
{{% /details %}}

{{% details title="Bitwuzla" closed="true" %}}
Requires MPFR >= 4.2.1 (`apt-get install libmpfr-dev` / `brew install mpfr`).
```sh
git clone --depth=1 --branch=0.9.0 https://github.com/bitwuzla/bitwuzla.git && cd bitwuzla && ./configure.py --prefix $PWD/../bitwuzla-release && cd build && meson install && cd ../..
```
{{% /details %}}

{{% details title="CVC4 (Linux only)" closed="true" %}}
```sh
pip3 install toml && git clone https://github.com/CVC4/CVC4.git && cd CVC4 && git reset --hard b826fc8ae95fc && ./contrib/get-antlr-3.4 && ./configure.sh --optimized --prefix=../cvc4 --static --no-static-binary && cd build && make -j4 && make install && cd ../..
```
{{% /details %}}

{{% details title="CVC5" closed="true" %}}
```sh
pip3 install toml && git clone https://github.com/CVC5/CVC5.git && cd CVC5 && git switch --detach cvc5-1.1.2 && ./configure.sh --prefix=../cvc5 --auto-download --static --no-static-binary && cd build && make -j4 && make install && cd ../..
```
{{% /details %}}

{{% details title="MathSAT" closed="true" %}}
```sh
# Linux
wget http://mathsat.fbk.eu/release/mathsat-5.5.4-linux-x86_64.tar.gz -O mathsat.tar.gz && tar xf mathsat.tar.gz && mv mathsat-5.5.4-linux-x86_64 mathsat
# macOS
wget http://mathsat.fbk.eu/release/mathsat-5.5.4-darwin-libcxx-x86_64.tar.gz -O mathsat.tar.gz && tar xf mathsat.tar.gz && mv mathsat-5.5.4-darwin-libcxx-x86_64 mathsat && ln -s /usr/local/include/gmp.h mathsat/include/gmp.h
```
{{% /details %}}

{{% details title="Yices (Linux)" closed="true" %}}
Yices needs a static GMP first:
```sh
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz && tar xf gmp-6.1.2.tar.xz && cd gmp-6.1.2 && ./configure --prefix $PWD/../gmp --disable-shared ABI=64 CFLAGS=-fPIC CPPFLAGS=-DPIC && make -j4 && make install && cd ..
git clone https://github.com/SRI-CSL/yices2.git && cd yices2 && git checkout Yices-2.6.4 && autoreconf -fi && ./configure --prefix $PWD/../yices --with-static-gmp=$PWD/../gmp/lib/libgmp.a && make -j9 && make static-lib && make install && cp ./build/x86_64-pc-linux-gnu-release/static_lib/libyices.a ../yices/lib && cd ..
```
{{% /details %}}

### Configure and build ESBMC

Pass the directories of the solvers you built. Drop the flags for any solver you
skipped.

```sh
cd esbmc && cmake -GNinja -Bbuild -DBUILD_TESTING=On -DENABLE_REGRESSION=On \
  $ESBMC_CLANG -DBUILD_STATIC=${ESBMC_STATIC:-ON} \
  -DBoolector_DIR=$PWD/../boolector-release -DZ3_DIR=$PWD/../z3 \
  -DENABLE_MATHSAT=ON -DMathsat_DIR=$PWD/../mathsat \
  -DENABLE_YICES=On -DYices_DIR=$PWD/../yices -DCVC4_DIR=$PWD/../cvc4 \
  -DGMP_DIR=$PWD/../gmp -DBitwuzla_DIR=$PWD/../bitwuzla-release \
  -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../release
ninja -C build && ninja -C build install
```

ESBMC is installed into the `release` folder. Add `-DCMAKE_BUILD_TYPE=Debug` to
enable ESBMC's internal assertions.

{{% /steps %}}

## Advanced

{{% details title="Build in a Docker container" closed="true" %}}
A minimal Debian-based image that builds ESBMC with Z3, letting
`-DDOWNLOAD_DEPENDENCIES=1` fetch LLVM/Clang and the libraries:

```dockerfile
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git bison flex \
      python3 libboost-all-dev g++-multilib \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth=1 https://github.com/esbmc/esbmc.git /esbmc
WORKDIR /esbmc
RUN cmake -GNinja -Bbuild -DDOWNLOAD_DEPENDENCIES=1 -DENABLE_Z3=1 \
    && ninja -C build
```

Build the image with `docker build -t esbmc .`; the binary is at
`/esbmc/build/src/esbmc/esbmc`.
{{% /details %}}

{{% details title="Shared (dynamic) builds" closed="true" %}}
A non-static ESBMC links against system libraries/solvers. Shared linking is the
default when CMake is invoked with `-DBUILD_STATIC=Off` (or the variable unset).

When Clang is built with `CLANG_LINK_CLANG_DYLIB=On`, ESBMC links the dynamic
`libclang-cpp` and does not bundle Clang's headers. Header bundling is controlled
by `CLANG_HEADERS_BUNDLED` (`On`/`Off`/`detect`); the default `detect` bundles
headers only for static Clang links. Not bundling speeds up source translation by
avoiding a temporary header-extraction directory, but ESBMC must be recompiled
when the system Clang is updated.
{{% /details %}}

{{% details title="CHERI-C support (experimental)" closed="true" %}}
CHERI-enabled ESBMC uses [CHERI Clang](https://github.com/CTSRD-CHERI/llvm-project)
(release 20210817, clang 13). Build it, then configure ESBMC against it:

```sh
wget https://github.com/CTSRD-CHERI/llvm-project/archive/refs/tags/cheri-rel-20210817.tar.gz
sudo apt-get install lld
tar xf cheri-rel-20210817.tar.gz && mkdir clang13 && cd llvm-project-cheri-rel-20210817 && mkdir build && cd build
cmake -GNinja -S ../llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS='llvm;clang' -DLLVM_INSTALL_BINUTILS_SYMLINKS=TRUE -DLLVM_ENABLE_LIBXML2=FALSE -DLLVM_ENABLE_ZLIB=FALSE '-DLLVM_TARGETS_TO_BUILD=AArch64;ARM;Mips;RISCV;X86;host' -DCMAKE_INSTALL_PREFIX=../../clang13
ninja && ninja install && cd ../..
ESBMC_CLANG=$(echo -D{LLVM,Clang}_DIR=$PWD/clang13)
```

A CHERI sysroot is needed for programs that use the C standard library. The
[cheribuild](https://github.com/CTSRD-CHERI/cheribuild) tool is the recommended
way to obtain one:

```sh
git clone https://github.com/CTSRD-CHERI/cheribuild.git && cd cheribuild && python3 cheribuild.py cheribsd-sdk-riscv64-purecap -d
```

Then configure ESBMC with:

```sh
-DESBMC_CHERI=On -DESBMC_CHERI_HYBRID_SYSROOT=<path> -DESBMC_CHERI_PURECAP_SYSROOT=<path>
```

e.g. `<path>` pointing at `$HOME/cheri/output/sdk/sysroot-riscv64-purecap`.
{{% /details %}}

{{% details title="Fuzzing targets" closed="true" %}}
ESBMC ships libFuzzer targets. They must be built with Clang, so configure ESBMC
to use it and enable fuzzing:

```sh
cmake -GNinja -Bbuild -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DENABLE_FUZZER=1
```

Passing the compiler options for the first time clears the CMake cache, so some
variables may need to be re-set.
{{% /details %}}
