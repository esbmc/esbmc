set (inputs limits.hs pthread.hs pthreadtypes.hs stdarg.hs stdbool.hs stddef.hs)

function(mangle_hs output)
  set(result "")
  foreach(in_f ${ARGN})
    file(RELATIVE_PATH out_f ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${in_f})
    string(REGEX REPLACE .hs .c out_f "${out_f}")
    set(out_f "${CMAKE_CURRENT_BINARY_DIR}/${out_f}")
    add_custom_command(OUTPUT ${out_f}
      COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py ${CMAKE_CURRENT_SOURCE_DIR}/${in_f} ${out_f}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${in_f}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Converting internal headers ${in_f} to data"
      VERBATIM
      )
    list(APPEND result ${out_f})
  endforeach()
  set(${output} "${result}" PARENT_SCOPE)
endfunction()

mangle_hs(mangled_files ${inputs})
add_library(c2gotoheaders ${mangled_files})
set_target_properties(c2gotoheaders PROPERTIES LINKER_LANGUAGE C)
