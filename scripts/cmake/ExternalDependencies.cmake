# Module to add dependencies that do not belong
# anywhere else

if(DOWNLOAD_DEPENDENCIES)
  include(FetchContent)

  # FMT
  fetchcontent_declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 12.1.0)
  fetchcontent_makeavailable(fmt)

  #nlohmann json
  fetchcontent_declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
  fetchcontent_makeavailable(json)
  # The pinned headers must beat any system copy for every target. Several
  # targets list ${Boost_INCLUDE_DIRS} (e.g. /opt/homebrew/include, which can
  # also carry a different nlohmann-json) in their own include list, which
  # precedes the nlohmann_json::nlohmann_json link-interface include; targets
  # then silently compile against the system version while the fetched one is
  # linked elsewhere, and the mixed inline-namespace ABIs
  # (nlohmann::json_abi_v3_11_3 vs v3_12_0) fail to link across libraries
  # (seen in unit/python-frontend). Prepending directory-wide enforces the
  # pin uniformly. Deliberately not SYSTEM: -isystem paths are searched
  # after every plain -I path, which would hand precedence back to the
  # system copy.
  include_directories(BEFORE "${json_SOURCE_DIR}/include")

  # yaml-cpp (pinned for reproducible builds; was tracking the default branch)
  fetchcontent_declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG yaml-cpp-0.9.0)
  fetchcontent_makeavailable(yaml-cpp)
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
  fetchcontent_declare(cheri_compressed_cap
    GIT_REPOSITORY https://github.com/CTSRD-CHERI/cheri-compressed-cap.git)
  fetchcontent_getproperties(cheri_compressed_cap)
  if(NOT cheri_compressed_cap_POPULATED)
    set(HAVE_UBSAN FALSE CACHE INTERNAL "")
    fetchcontent_populate(cheri_compressed_cap)
    add_subdirectory(${cheri_compressed_cap_SOURCE_DIR}
                     ${cheri_compressed_cap_BINARY_DIR}
                     EXCLUDE_FROM_ALL)
  endif()

  if(ESBMC_CHERI AND DOWNLOAD_DEPENDENCIES AND ("${ESBMC_CHERI_HYBRID_SYSROOT}" STREQUAL ""))
    fetchcontent_declare(cheri_sysroot
     URL https://github.com/XLiZHI/esbmc/releases/download/v17/sysroot-riscv64-purecap.zip)
    fetchcontent_makeavailable(cheri_sysroot)

    set(ESBMC_CHERI_HYBRID_SYSROOT ${cheri_sysroot_SOURCE_DIR})
    set(ESBMC_CHERI_PURECAP_SYSROOT ${cheri_sysroot_SOURCE_DIR})
  endif()

  # CHERI Clang AST: ignore other frontend
  unset(ENABLE_PYTHON_FRONTEND CACHE)
  unset(ENABLE_SOLIDITY_FRONTEND CACHE)
  unset(ENABLE_JIMPLE_FRONTEND CACHE)
endif()
