# Module to add dependencies that do not belong
# anywhere else

include(FetchContent)
# FMT
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.0)
FetchContent_MakeAvailable(fmt)

#nlohmann json
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.10.3)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

if(ESBMC_CHERI_CLANG)
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
endif()
