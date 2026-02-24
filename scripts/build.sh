#!/usr/bin/env bash

set -euo pipefail

on_error() {
  local line="$1"
  local cmd="$2"
  echo "[x] build.sh failed at line ${line}: ${cmd}" >&2
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

OS="$(uname -s)"
ARCH="$(uname -m)"

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  SUDO="sudo"
fi

# Set arguments that should be available for every OS.
BASE_ARGS="\
    -DDOWNLOAD_DEPENDENCIES=On \
    -GNinja \
    -DENABLE_CSMITH=On \
    -DBUILD_TESTING=On \
    -DENABLE_REGRESSION=On \
    -DENABLE_SOLIDITY_FRONTEND=On \
    -DENABLE_JIMPLE_FRONTEND=On \
    -DENABLE_PYTHON_FRONTEND=On \
    -DCMAKE_INSTALL_PREFIX:PATH=$ROOT_DIR/release \
"

SOLVER_FLAGS="\
    -DENABLE_BOOLECTOR=On \
    -DENABLE_YICES=Off \
    -DENABLE_BITWUZLA=On \
    -DENABLE_GOTO_CONTRACTOR=On \
    -DACADEMIC_BUILD=Off \
"

COMPILER_ARGS=''

STATIC=""
CLANG_VERSION=16
MIN_MACOS_CLANG_VERSION=17

if [[ "$OS" == "Darwin" ]]; then
  CLANG_VERSION="$MIN_MACOS_CLANG_VERSION"
fi

GMP_VERSION="6.3.0"
GMP_TARBALL="gmp-${GMP_VERSION}.tar.xz"
DEPS_CACHE_DIR="$ROOT_DIR/.deps"
GMP_ARCHIVE_PATH="$DEPS_CACHE_DIR/$GMP_TARBALL"

PLATFORM_CONFIGURED=0
FETCH_DONE=0

PACKAGE_INSTALL_CMD=()

error() {
  echo "error: $*" >&2
  exit 1
}

log() {
  echo "[~] $*"
}

validate_clang_version() {
  if ! [[ "$CLANG_VERSION" =~ ^[0-9]+$ ]]; then
    error "invalid clang version '$CLANG_VERSION': expected numeric major version"
  fi

  if [[ "$OS" == "Darwin" ]] && (( CLANG_VERSION < MIN_MACOS_CLANG_VERSION )); then
    error "macOS requires llvm/clang >= ${MIN_MACOS_CLANG_VERSION}; got $CLANG_VERSION"
  fi
}

run_with_sudo() {
  if [[ -n "$SUDO" ]]; then
    "$SUDO" "$@"
  else
    "$@"
  fi
}

ensure_build_dir() {
  mkdir -p build
}

configure_project() {
  ensure_build_dir
  cd build
  printf "Running CMake:"
  printf " '%s'" $COMPILER_ARGS cmake .. $BASE_ARGS $SOLVER_FLAGS -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  echo
  $COMPILER_ARGS cmake .. $BASE_ARGS $SOLVER_FLAGS -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  cd "$ROOT_DIR"
}

check_configured_build() {
  [[ -f "$ROOT_DIR/build/CMakeCache.txt" ]] || error "build is not configured. Run '$0 deps [flags]' first"
}

download_file_with_fallback() {
  local output="$1"
  shift
  local url

  for url in "$@"; do
    echo "Trying $url ..."
    if command -v wget >/dev/null 2>&1; then
      if wget -q --show-progress -O "$output" "$url"; then
        return 0
      fi
    fi
  done

  return 1
}

fetch_gmp_source() {
  mkdir -p "$DEPS_CACHE_DIR"

  if [[ -f "$GMP_ARCHIVE_PATH" ]]; then
    log "Using cached $GMP_TARBALL"
    return
  fi

  log "Fetching $GMP_TARBALL"
  download_file_with_fallback \
    "$GMP_ARCHIVE_PATH" \
    "https://ftp.gnu.org/gnu/gmp/$GMP_TARBALL" \
    "https://mirrors.kernel.org/gnu/gmp/$GMP_TARBALL" \
    "https://ftpmirror.gnu.org/gmp/$GMP_TARBALL" \
    || error "failed to download $GMP_TARBALL"
}

# Configure flags once per invocation.
prepare_platform_config() {
  if [[ "$PLATFORM_CONFIGURED" -eq 1 ]]; then
    return
  fi

  case "$OS" in
    Linux)
      if [[ -z "$STATIC" ]]; then
        STATIC=ON
      fi

      if [[ "$STATIC" == "OFF" ]]; then
        BASE_ARGS="$BASE_ARGS \
            -DClang_DIR=/usr/lib/cmake/clang-$CLANG_VERSION \
            -DLLVM_DIR=/usr/lib/llvm-$CLANG_VERSION/lib/cmake/llvm \
            -DZ3_DIR=/usr \
        "
        log "Configuring shared Ubuntu build with Clang-$CLANG_VERSION frontend"
      else
        log "Configuring static Ubuntu build"
      fi

      BASE_ARGS="$BASE_ARGS -DBUILD_STATIC=$STATIC"
      SOLVER_FLAGS="$SOLVER_FLAGS -DENABLE_Z3=ON -DENABLE_CVC5=On"

      if [[ "$ARCH" == "aarch64" ]]; then
        log "Detected ARM64 Linux"
        SOLVER_FLAGS="$SOLVER_FLAGS \
            -DENABLE_GOTO_CONTRACTOR=OFF \
            -DENABLE_CVC5=Off \
        "
      fi
      ;;

    Darwin)
      if [[ -z "$STATIC" ]]; then
        STATIC=OFF
      fi
      if [[ "$STATIC" == "ON" ]]; then
        error "static macOS build is currently not supported"
      fi
      log "Configuring macOS build with llvm/clang ${CLANG_VERSION}"

      BASE_ARGS="$BASE_ARGS \
        -DLLVM_DIR=/opt/homebrew/opt/llvm@$CLANG_VERSION \
        -DClang_DIR=/opt/homebrew/opt/llvm@$CLANG_VERSION \
        -DC2GOTO_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_INSTALL_PREFIX:PATH=$ROOT_DIR/release \
      "

      SOLVER_FLAGS="$SOLVER_FLAGS -DENABLE_GOTO_CONTRACTOR=OFF -DENABLE_Z3=ON"
      ;;

    *)
      error "unsupported OS '$OS'"
      ;;
  esac

  PLATFORM_CONFIGURED=1
}

collect_ubuntu_packages() {
  prepare_platform_config

  UBUNTU_PACKAGES=(
    python-is-python3
    csmith
    python3
    libbz2-dev
    liblzma-dev
    git
    unzip
    wget
    curl
    libcsmith-dev
    gperf
    cmake
    bison
    flex
    linux-libc-dev
    libboost-date-time-dev
    libboost-program-options-dev
    libboost-iostreams-dev
    libboost-system-dev
    libboost-filesystem-dev
    ninja-build
    python3-setuptools
    libncurses-dev
    python3-pip
    python3-toml
    default-jdk
    tar
    xz-utils
  )

  if [[ "$ARCH" != "aarch64" ]]; then
    UBUNTU_PACKAGES+=(g++-multilib)
  else
    log "Skipping g++-multilib on aarch64"
  fi

  if [[ "$STATIC" == "OFF" ]]; then
    UBUNTU_PACKAGES+=(
      "llvm-$CLANG_VERSION-dev"
      "libclang-$CLANG_VERSION-dev"
      "libclang-cpp${CLANG_VERSION}-dev"
      libz3-dev
    )
  fi
}

collect_macos_formulae() {
  prepare_platform_config

  MACOS_FORMULAE=(
    cmake
    z3
    gmp
    csmith
    boost
    ninja
    python@3.12
    automake
    bison
    flex
    "llvm@$CLANG_VERSION"
  )
}

set_linux_install_command_for_missing() {
  local missing=("$@")
  PACKAGE_INSTALL_CMD=(apt-get install -y "${missing[@]}")
  if [[ -n "$SUDO" ]]; then
    PACKAGE_INSTALL_CMD=("$SUDO" "${PACKAGE_INSTALL_CMD[@]}")
  fi
}

set_macos_install_command_for_missing() {
  local missing=("$@")
  PACKAGE_INSTALL_CMD=(brew install "${missing[@]}")
}

check_ubuntu_packages() {
  command -v apt-get >/dev/null 2>&1 || error "unsupported Linux distribution: apt-get not found"

  collect_ubuntu_packages

  local missing=()
  local pkg
  for pkg in "${UBUNTU_PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
      missing+=("$pkg")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    PACKAGE_INSTALL_CMD=()
    log "OK: Ubuntu packages required for build are installed"
    return 0
  fi

  set_linux_install_command_for_missing "${missing[@]}"
  echo "[!] Missing Ubuntu packages: ${missing[*]}"
  echo "[?] Install with: ${PACKAGE_INSTALL_CMD[*]}"
  return 1
}

check_macos_formulae() {
  command -v brew >/dev/null 2>&1 || error "Homebrew is required"

  collect_macos_formulae

  local missing=()
  local formula
  for formula in "${MACOS_FORMULAE[@]}"; do
    if ! brew list --versions "$formula" >/dev/null 2>&1; then
      missing+=("$formula")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    PACKAGE_INSTALL_CMD=()
    log "OK: macOS brew formulae required for build are installed"
    return 0
  fi

  set_macos_install_command_for_missing "${missing[@]}"
  echo "[!] Missing brew formulae: ${missing[*]}"
  echo "[?] Install with: ${PACKAGE_INSTALL_CMD[*]}"
  return 1
}

check_system_dependencies() {
  case "$OS" in
    Linux)
      check_ubuntu_packages
      ;;
    Darwin)
      check_macos_formulae
      ;;
    *)
      error "unsupported OS '$OS'"
      ;;
  esac
}

install_system_dependencies() {
  if check_system_dependencies; then
    return
  fi

  [[ ${#PACKAGE_INSTALL_CMD[@]} -gt 0 ]] || error "no package installation command prepared"
  log "Installing missing system dependencies"
  "${PACKAGE_INSTALL_CMD[@]}"
}

install_gmp_linux() {
#   if [[ "$ARCH" == "aarch64" ]]; then
#     log "Skipping GMP source build on ARM64 Linux"
#     return
#   fi

  if command -v pkg-config >/dev/null 2>&1; then
    local installed_version
    installed_version="$(pkg-config --modversion gmp 2>/dev/null || true)"
    if [[ "$installed_version" == "$GMP_VERSION" ]]; then
      log "GMP $GMP_VERSION already installed"
      return
    fi
  fi

  fetch_gmp_source

  log "Installing GMP $GMP_VERSION from source"
  local build_root
  build_root="$(mktemp -d)"
  tar -xf "$GMP_ARCHIVE_PATH" -C "$build_root"
  cd "$build_root/gmp-$GMP_VERSION"
  ./configure --prefix=/usr/local --enable-cxx --enable-static
  make -j"$(nproc)"
  run_with_sudo make install
  run_with_sudo ldconfig || true
  cd "$ROOT_DIR"
  rm -rf "$build_root"
}

install_python_deps_linux() {
  log "Installing Python dependencies"
  python3 -m pip install --user meson ast2json mypy pyparsing toml tomli
  meson --version
}

install_python_deps_macos() {
  log "Installing Python dependencies"

  local py312
  local py312_bin
  py312="$(brew --prefix python@3.12)/bin/python3.12"
  py312_bin="$(brew --prefix python@3.12)/bin"

  export Python3_EXECUTABLE="$py312"

  # Ensure python3 points to python3.12 in brew's Python 3.12 bin directory.
  if [[ ! -f "$py312_bin/python3" ]]; then
    ln -sf python3.12 "$py312_bin/python3"
  fi

  export PATH="$py312_bin:$HOME/Library/Python/3.12/bin:$PATH"

  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    pip install meson ast2json mypy pyparsing toml tomli jira
  else
    "$py312" -m pip install --user --break-system-packages meson ast2json mypy pyparsing toml tomli jira
  fi

  meson --version

  SOLVER_FLAGS="$SOLVER_FLAGS -DZ3_DIR=$(brew --prefix z3)"
  BASE_ARGS="$BASE_ARGS -DPython3_EXECUTABLE=$Python3_EXECUTABLE"
}

run_fetch() {
  if [[ "$FETCH_DONE" -eq 1 ]]; then
    return
  fi

  case "$OS" in
    Linux)
      run_with_sudo apt-get update
      fetch_gmp_source
      ;;
    Darwin)
      brew update
      ;;
    *)
      error "unsupported OS '$OS'"
      ;;
  esac

  FETCH_DONE=1
}

run_check() {
  log "Checking dependency prerequisites"
  check_system_dependencies

  if [[ -f "$ROOT_DIR/build/CMakeCache.txt" ]]; then
    log "Configured build directory detected"
  else
    log "No configured build directory detected (expected before first install)"
  fi
}

run_install_deps() {
  prepare_platform_config

  # Keep install independent from whether fetch was called explicitly.
  if [[ "$FETCH_DONE" -eq 0 ]]; then
    run_fetch
  fi

  install_system_dependencies

  case "$OS" in
    Linux)
      install_gmp_linux
      install_python_deps_linux
      export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
      export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
      export CMAKE_PREFIX_PATH="/usr/local:${CMAKE_PREFIX_PATH:-}"
      ;;
    Darwin)
      install_python_deps_macos
      ;;
    *)
      error "unsupported OS '$OS'"
      ;;
  esac

  configure_project
}

run_build_esbmc() {
  check_configured_build
  cd build
  cmake --build .
  cd "$ROOT_DIR"
}

run_install_esbmc() {
  check_configured_build
  cd build
  cmake --install .
  cd "$ROOT_DIR"

  case "$OS" in
    Linux)
      log "No further steps needed for Ubuntu"
      ;;
    Darwin)
      log "No further steps needed for macOS"
      ;;
    *)
      error "unsupported OS '$OS'"
      ;;
  esac
}

usage() {
  cat <<USAGE
$0 [-OPTS] [deps] [build] [install]

Options [defaults]:
  -h         display this help message
  -b BTYPE   set cmake build type to BTYPE [RelWithDebInfo]
  -s STYPE   enable sanitizer STYPE and compile with clang [disabled]
  -e ON|OFF  enable/disable -Werror [OFF]
  -r ON|OFF  enable/disable 'benchbringup' [OFF]
  -d         enable debug output for this script and c2goto
  -S ON|OFF  enable/disable static build [ON for Ubuntu, OFF for macOS]
  -c VERS    use packaged clang-VERS [16 on Linux, >=17 required on macOS]
  -C         build an SV-COMP version [disabled]
  -B ON|OFF  enable/disable esbmc bundled libc [ON]
  -x ON|OFF  enable/disable esbmc cheri [OFF]

Commands:
  fetch-deps         fetch dependency metadata and source archives [internal]
  check-deps         check required system dependencies and print install command when missing [internal]
  install-deps       install dependencies and configure the build directory [internal]
  deps               run 'fetch-deps', 'install-deps' and 'check-deps' in sequence
  build              build ESBMC (does not install dependencies)
  install            install ESBMC from the configured build directory

Default behavior (when no command is given): deps build install

Needs to be executed from the top-level directory of ESBMC's source tree.
Supported environments are: Ubuntu-22.04 and macOS.
USAGE
}

# Setup build flags (release, debug, sanitizer, ...)
while getopts "hb:s:e:r:dS:c:CB:x:" flag; do
  case "$flag" in
    h)
      usage
      exit 0
      ;;
    b)
      BASE_ARGS="$BASE_ARGS -DCMAKE_BUILD_TYPE=${OPTARG}"
      ;;
    s)
      BASE_ARGS="$BASE_ARGS -DSANITIZER_TYPE=${OPTARG}"
      COMPILER_ARGS="$COMPILER_ARGS CC=clang CXX=clang++"
      ;;
    e)
      BASE_ARGS="$BASE_ARGS -DENABLE_WERROR=${OPTARG}"
      ;;
    r)
      BASE_ARGS="$BASE_ARGS -DBENCHBRINGUP=${OPTARG}"
      ;;
    d)
      set -x
      export ESBMC_OPTS='--verbosity 9'
      ;;
    S)
      STATIC="$OPTARG"
      ;;
    c)
      CLANG_VERSION="$OPTARG"
      ;;
    C)
      BASE_ARGS="$BASE_ARGS -DESBMC_SVCOMP=ON"
      SOLVER_FLAGS="\
          -DENABLE_BOOLECTOR=On \
          -DENABLE_YICES=On \
          -DENABLE_CVC4=OFF \
          -DENABLE_BITWUZLA=On \
          -DENABLE_Z3=On \
          -DENABLE_MATHSAT=ON \
          -DENABLE_GOTO_CONTRACTOR=OFF \
          -DACADEMIC_BUILD=ON"
      ;;
    B)
      BASE_ARGS="$BASE_ARGS -DESBMC_BUNDLE_LIBC=$OPTARG"
      ;;
    x)
      BASE_ARGS="$BASE_ARGS \
          -DENABLE_SOLIDITY_FRONTEND=OFF \
          -DENABLE_JIMPLE_FRONTEND=OFF \
          -DENABLE_PYTHON_FRONTEND=OFF \
          -DESBMC_CHERI=ON"
      SOLVER_FLAGS="\
          -DENABLE_BOOLECTOR=On \
          -DENABLE_Z3=On"
      ;;
    *)
      error "invalid option"
      ;;
  esac
done
shift $((OPTIND - 1))

validate_clang_version

ACTIONS=("$@")
if [[ ${#ACTIONS[@]} -eq 0 ]]; then
  ACTIONS=(deps build install)
fi

for action in "${ACTIONS[@]}"; do
  case "$action" in
    deps)
      run_fetch
      run_install_deps
      run_check
      ;;
    fetch-deps)
      run_fetch
      ;;
    check-deps)
      run_check
      ;;
    install-deps)
      run_install_deps
      ;;
    build)
      run_build_esbmc
      ;;
    install)
      run_install_esbmc
      ;;
    help)
      usage
      exit 0
      ;;
    *)
      log "unknown command: $action"
      usage
      exit 1
      ;;
  esac
done
