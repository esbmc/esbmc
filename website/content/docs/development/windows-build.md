---
title: Windows Build
---

This page contains info on building ESBMC on Windows, for other OS's use the [BUILDING.md](https://github.com/esbmc/esbmc/blob/master/BUILDING.md) document.

# Common Build

First you need to Download and configure the dependencies.

- If you downloaded the repository or cloned be sure that all line endings are following LF notation. CRLF will not work.
- [Visual Studio](https://visualstudio.microsoft.com/). When selecting the workload go for the C++ Desktop Development. **Note:**
 If you are installing a non-english version, install the English language pack.
- [Chocolatey](https://chocolatey.org/)
- [vcpkg](https://github.com/microsoft/vcpkg). Follow the instructions and integrate it to your system if you want to use Visual Studio or if you want to use CMake directly get the toolchain command (vcpkg docs).
- [Z3](https://github.com/Z3Prover/z3/releases)
- Prebuilt LLVM. You can manually compile it (instructions bellow) or use [this](https://www.dropbox.com/s/z1gyschfa46yj6e/clang.zip?dl=1). Note that the official releases won't come with the needed CMake modules. 

### Installing libraries
 
- `vcpkg.exe install boost-filesystem:x64-Windows boost-date-time:x64-windows boost-test:x64-windows boost-multi-index:x64-windows boost-crc:x64-windows boost-property-tree:x64-windows boost-uuid:x64-windows`
- `chocolatey install winflexbison gnuwin32-coreutils.portable`

### Building

1. Open Visual Studio and import the CMake project.
1. Open CMake configuration window
1. Mark the Configuration type as Release.
1. If you know what you are doing you can select any toolchain you want (reflecting into vcpkg). If you followed the instructions so far, you should use msvc_x64_x64
1. In cmake configuration flags you should add the path to where you downloaded Z3 and LLVM e.g `-DLLVM_DIR=C:\\Deps\\llvm9d -DClang_DIR=C:\\Deps\\llvm9d -DZ3_DIR=C:\\Deps\\z3`
1. You should also add the `-DBOOST_DLL_FILE` to the downloaded file, if you want to use yours, it should be available at `<VCPKG_ROOG>\buildtrees\boost-filesystem\x64-windows-rel\stage\lib` 
1. On vcpkg-target-triples mark as `x64-windows`. If you know what you are doing you should be able to use any triplet you want.

### Linux Build on Windows

By using Windows [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) you can build and install ESBMC as if you were using an `Ubuntu`.

# Building LLVM From source

The building process was shamelessly taken from Zig repository: https://github.com/ziglang/zig/wiki/How-to-build-LLVM,-libclang,-and-liblld-from-source#windows

### Setup

Install [CMake](https://cmake.org/), version 3.17 or newer.

[Download llvm, and clang](http://releases.llvm.org/download.html#14.0.0) The downloads from llvm lead to the github release pages, where the source's will be listed as : `llvm-14.X.X.src.tar.xz`, `clang-14.X.X.src.tar.xz`. Unzip each to their own directory. Ensure no directories have spaces in them. For example:

 * `C:\Users\Andy\llvm-14.0.6.src`
 * `C:\Users\Andy\clang-14.0.6.src`

Install [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019). Be sure to select "C++ build tools" when prompted.
 * You **must** additionally check the optional component labeled **C++ ATL for v142 build tools**. As this won't be supplied by a default installation of Visual Studio.
 * Full list of supported MSVC versions:
   - 2017 (version 15.8)
   - 2019 (version 16)
   - 2022 (version 17)

Install [Python 3.9.4](https://www.python.org). Tick the box to add python to your PATH environment variable.

### LLVM

Using the start menu, run **x64 Native Tools Command Prompt for VS 2019** and execute these commands, replacing `C:\Users\Andy` with the correct value. Here is listed a brief explanation of each of the CMake parameters we pass when configuring the build 

- `-Thost=x64` : Sets the windows toolset to use 64 bit mode.
- `-A x64` : Make the build target 64 bit .
- `-G "Visual Studio 16 2019"` : Specifies to generate a 2019 Visual Studio project, the best supported version.
- `-DCMAKE_INSTALL_PREFIX=""` : Path that llvm components will being installed into by the install project.
- `-DCMAKE_PREFIX_PATH=""` : Path that CMake will look into first when trying to locate dependencies, should be the same place as the install prefix. This will ensure that clang and lld will use your newly built llvm libraries.
- `-DLLVM_ENABLE_ZLIB=OFF` : Don't build llvm with ZLib support as it's not required and will disrupt the target dependencies for components linking against llvm. This only has to be passed when building llvm, as this option will be saved into the config headers.
- `-DCMAKE_BUILD_TYPE=Release` : Build llvm and components in release mode.
- `-DCMAKE_BUILD_TYPE=Debug` : Build llvm and components in debug mode.
- `-DLLVM_USE_CRT_RELEASE=MT` : Which C runtime should llvm use during release builds.
- `-DLLVM_USE_CRT_DEBUG=MTd` : Make llvm use the debug version of the runtime in debug builds.

#### Release Mode

```bat
mkdir C:\Users\Andy\llvm-14.0.6.src\build-release
cd C:\Users\Andy\llvm-14.0.6.src\build-release
"c:\Program Files\CMake\bin\cmake.exe" .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=C:\Users\Andy\llvm+clang+lld-14.0.6-x86_64-windows-msvc-release-mt -DCMAKE_PREFIX_PATH=C:\Users\Andy\llvm+clang+lld-14.0.6-x86_64-windows-msvc-release-mt -
DLLVM_ENABLE_ZLIB=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_USE_CRT_RELEASE=MT
msbuild /m -p:Configuration=Release INSTALL.vcxproj
```

#### Debug Mode

```bat
mkdir C:\Users\Andy\llvm-14.0.6.src\build-debug
cd C:\Users\Andy\llvm-14.0.6.src\build-debug
"c:\Program Files\CMake\bin\cmake.exe" .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=C:\Users\andy\llvm+clang+lld-14.0.6-x86_64-windows-msvc-debug -
DLLVM_ENABLE_ZLIB=OFF -DCMAKE_PREFIX_PATH=C:\Users\andy\llvm+clang+lld-14.0.6-x86_64-windows-msvc-debug -DCMAKE_BUILD_TYPE=Debug -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="AVR" -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_USE_CRT_DEBUG=MTd
msbuild /m INSTALL.vcxproj
```

### Clang

Using the start menu, run **x64 Native Tools Command Prompt for VS 2019** and execute these commands, replacing `C:\Users\Andy` with the correct value.

#### Release Mode

```bat
mkdir C:\Users\Andy\clang-14.0.6.src\build-release
cd C:\Users\Andy\clang-14.0.6.src\build-release
"c:\Program Files\CMake\bin\cmake.exe" .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=C:\Users\Andy\llvm+clang+lld-14.0.6-x86_64-windows-msvc-release-mt -DCMAKE_PREFIX_PATH=C:\Users\Andy\llvm+clang+lld-14.0.6-x86_64-windows-msvc-release-mt -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_CRT_RELEASE=MT
msbuild /m -p:Configuration=Release INSTALL.vcxproj
```

#### Debug Mode

```bat
mkdir C:\Users\Andy\clang-14.0.6.src\build-debug
cd C:\Users\Andy\clang-14.0.6.src\build-debug
"c:\Program Files\CMake\bin\cmake.exe" .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=C:\Users\andy\llvm+clang+lld-14.0.6-x86_64-windows-msvc-debug -DCMAKE_PREFIX_PATH=C:\Users\andy\llvm+clang+lld-14.0.6-x86_64-windows-msvc-debug -DCMAKE_BUILD_TYPE=Debug -DLLVM_USE_CRT_DEBUG=MTd
msbuild /m INSTALL.vcxproj
```

## Posix

This guide will get you both a Debug build of LLVM, and/or a Release build of LLVM.
It intentionally does not require privileged access, using a prefix inside your home
directory instead of a global installation. This might be useful for doing MinGW builds!

### Release

This is the generally recommended approach.

```
cd ~/Downloads
git clone --depth 1 --branch release/15.x https://github.com/llvm/llvm-project llvm-project-15
cd llvm-project-15
git checkout release/15.x
mkdir build-release
cd build-release
cmake ../llvm \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/llvm15-release \
  -DCMAKE_PREFIX_PATH=$HOME/local/llvm15-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="lld;clang" \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -G Ninja \
  -DLLVM_PARALLEL_LINK_JOBS=1
ninja install
```

### Debug

This is occasionally needed when debugging Zig's LLVM backend.

```
# Skip this step if you already did it for Release above.
cd ~/Downloads
git clone --depth 1 --branch release/15.x https://github.com/llvm/llvm-project llvm-project-15
cd llvm-project-15
git checkout release/15.x

# LLVM
cd llvm
mkdir build-debug
cd build-debug
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local/llvm15-debug -DCMAKE_PREFIX_PATH=$HOME/local/llvm15-debug -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_TERMINFO=OFF -G Ninja -DLLVM_PARALLEL_LINK_JOBS=1
cmake --build .
ninja install
cd ../..

# Clang
cd clang
mkdir build-debug
cd build-debug
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local/llvm15-debug -DCMAKE_PREFIX_PATH=$HOME/local/llvm15-debug -DCMAKE_BUILD_TYPE=Release  -G Ninja -DLLVM_PARALLEL_LINK_JOBS=1
cmake --build .
ninja install
cd ../..
```
