# Module to add dependencies that do not belong
# anywhere else

if(DOWNLOAD_DEPENDENCIES)
  include(FetchContent)

  # FMT
  FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 12.1.0)
  FetchContent_MakeAvailable(fmt)

  # nlohmann json
  FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(json)

  # yaml-cpp
  FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git)
  FetchContent_MakeAvailable(yaml-cpp)
else()
  # Use system-installed libraries (required for distro packaging).
  find_package(fmt REQUIRED)
  if(TARGET fmt AND NOT TARGET fmt::fmt)
    add_library(fmt::fmt ALIAS fmt)
  endif()

  find_package(nlohmann_json 3.2.0 REQUIRED)
  if(TARGET nlohmann_json AND NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
  endif()

  find_package(yaml-cpp REQUIRED)
  if(TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
    add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
  endif()
endif()

if(ESBMC_CHERI_CLANG)
  include(FetchContent)
  FetchContent_Declare(cheri_compressed_cap
    GIT_REPOSITORY https://github.com/CTSRD-CHERI/cheri-compressed-cap.git)
  FetchContent_GetProperties(cheri_compressed_cap)
  if(NOT cheri_compressed_cap_POPULATED)
    set(HAVE_UBSAN FALSE CACHE INTERNAL "")
    FetchContent_Populate(cheri_compressed_cap)
    add_subdirectory(${cheri_compressed_cap_SOURCE_DIR}
                     ${cheri_compressed_cap_BINARY_DIR}
                     EXCLUDE_FROM_ALL)
  endif()

  if(ESBMC_CHERI AND DOWNLOAD_DEPENDENCIES AND ("${ESBMC_CHERI_HYBRID_SYSROOT}" STREQUAL ""))
    FetchContent_Declare(cheri_sysroot
     URL https://github.com/XLiZHI/esbmc/releases/download/v17/sysroot-riscv64-purecap.zip)
    FetchContent_MakeAvailable(cheri_sysroot)

    set(ESBMC_CHERI_HYBRID_SYSROOT ${cheri_sysroot_SOURCE_DIR})
    set(ESBMC_CHERI_PURECAP_SYSROOT ${cheri_sysroot_SOURCE_DIR})
  endif()

  # CHERI Clang AST: ignore other frontend
  unset(ENABLE_PYTHON_FRONTEND CACHE)
  unset(ENABLE_SOLIDITY_FRONTEND CACHE)
  unset(ENABLE_JIMPLE_FRONTEND CACHE)
endif()
