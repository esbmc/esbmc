
if (ESBMC_CLANG_HEADERS_BUNDLED)
  message(STATUS "Bundling clang headers from: ${ESBMC_CLANG_HEADER_DIR}")
  mangle(mangled_c ${ESBMC_CLANG_HEADER_DIR} WILDCARD "*.h"
         MACRO ESBMC_FLAIL
         ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/cheaders.h
         LIST_DIRECTORIES FALSE)

  add_library(cheaders ${mangled_c})
  set_target_properties(cheaders PROPERTIES LINKER_LANGUAGE C
                                 PRIVATE_HEADER ${CMAKE_CURRENT_BINARY_DIR}/cheaders.h)

  set(cheaders_lib cheaders PARENT_SCOPE)
else()
  set(cheaders_lib "" PARENT_SCOPE)
endif()
