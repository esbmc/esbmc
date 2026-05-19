# Module to add dependencies that do not belong
# anywhere else

include(FetchContent)
# FMT
fetchcontent_declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 12.1.0)
fetchcontent_makeavailable(fmt)

#nlohmann json
fetchcontent_declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
fetchcontent_makeavailable(json)

# yaml-cpp
fetchcontent_declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git)
fetchcontent_makeavailable(yaml-cpp)

if(ESBMC_CHERI_CLANG)
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
