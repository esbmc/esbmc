
function(mangle_hs output_c output_h dir)
  set(result_c "")
  set(result_h "")
  file(GLOB inputs LIST_DIRECTORIES FALSE CONFIGURE_DEPENDS ${dir}/*.h)
  foreach(in_f ${inputs})
    file(RELATIVE_PATH out_f ${dir} ${in_f})
    string(REGEX REPLACE "(.*)\\.h" "${CMAKE_CURRENT_BINARY_DIR}/\\1" out "${out_f}")
    add_custom_command(OUTPUT ${out}.c ${out}.h
      COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py
        --macro ESBMC_FLAIL --header ${out}.h
        ${in_f}
        ${out}.c
      DEPENDS ${in_f}
              ${CMAKE_SOURCE_DIR}/scripts/flail.py
              ${CMAKE_CURRENT_BINARY_DIR}/cheaders.h
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Converting clang headers ${in_f} to data"
      VERBATIM
      )
    list(APPEND result_c ${out}.c)
    list(APPEND result_h ${out}.h)
  endforeach()
  set(${output_c} "${result_c}" PARENT_SCOPE)
  set(${output_h} "${result_h}" PARENT_SCOPE)
endfunction()

if (ESBMC_CLANG_HEADER_DIR AND NOT CLANG_HEADERS_BUNDLED)
  set(cheaders_lib "" PARENT_SCOPE)
else()
  message(STATUS "Bundling clang headers from: ${ESBMC_CLANG_HEADER_DIR}")
  mangle_hs(mangled_c mangled_h ${ESBMC_CLANG_HEADER_DIR})
  unset(ESBMC_CLANG_HEADER_DIR)

  list(TRANSFORM mangled_h REPLACE "(.+)" "#include \"\\1\"" OUTPUT_VARIABLE includes)
  list(JOIN includes "\n" includes)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cheaders.h ${includes})

  add_library(cheaders ${mangled_c})
  set_target_properties(cheaders PROPERTIES LINKER_LANGUAGE C
                                 PRIVATE_HEADER ${CMAKE_CURRENT_BINARY_DIR}/cheaders.h)

  set(cheaders_lib cheaders PARENT_SCOPE)
endif()
