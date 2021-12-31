add_executable(c2goto c2goto.cpp cprover_blank_library.cpp)
target_include_directories(c2goto
    PRIVATE ${Boost_INCLUDE_DIRS}
    )
  
target_link_libraries(c2goto 
  clangcfrontend gotoprograms langapi util_esbmc irep2
  default_message ${Boost_LIBRARIES})
add_subdirectory (headers)

if(DEFINED C2GOTO_INCLUDE_DIR)
  set(multiarch "${C2GOTO_INCLUDE_DIR}")
else()
  set(multiarch "/usr/include/${CMAKE_LIBRARY_ARCHITECTURE}")
endif()

set (clib32_config clib32.c --32 --fixedbv -D__ESBMC_FIXEDBV)
set (clib64_config clib64.c --64 --fixedbv -D__ESBMC_FIXEDBV)
set (clib32f_config clib32_fp.c --32 --floatbv)
set (clib64f_config clib64_fp.c --64 --floatbv)

set (inputs clib32_config clib64_config clib32f_config clib64f_config)

function(mangle_clib output)
  set(result "")
   file(GLOB c2goto_library_files
    "${CMAKE_CURRENT_SOURCE_DIR}/library/*.c")
  file(GLOB c2goto_libm_files
    "${CMAKE_CURRENT_SOURCE_DIR}/library/libm/*.c")
  foreach(in_f ${ARGN})
    set (in_f "${${in_f}}")
    list(GET in_f 0 in_file)
    list(SUBLIST in_f 1 -1 in_flags)
    file(RELATIVE_PATH out_file ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${in_file})
    string(REGEX REPLACE .c .goto out_goto "${in_file}")
    string(REGEX REPLACE .c "" barename "${in_file}")
    set(out_goto "${CMAKE_CURRENT_BINARY_DIR}/${out_goto}")
    set(out_file "${CMAKE_CURRENT_BINARY_DIR}/${out_file}")

    if(ENABLE_LIBM)
      set(inputs_c ${c2goto_library_files} ${c2goto_libm_files})
    else()
      set(inputs_c ${c2goto_library_files})
    endif()
    set(CMD c2goto -I ${multiarch} ${OS_C2GOTO_FLAGS} ${inputs_c} ${in_flags} --output ${out_goto})
    add_custom_command(OUTPUT ${out_goto}
      COMMAND ${CMD}
      DEPENDS c2goto ${c2goto_library_files} ${c2goto_libm_files}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating libc model ${out_goto}"
      VERBATIM
      )

    add_custom_command(OUTPUT ${out_file}
      COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py ${out_goto} ${out_file}
      DEPENDS ${out_goto} ${CMAKE_SOURCE_DIR}/scripts/flail.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Converting libc model ${out_goto} to data"
      VERBATIM
      )

    list(APPEND result ${out_file})
  endforeach()
  set(${output} "${result}" PARENT_SCOPE)
endfunction()

message(STATUS "Multiarch include for c2goto: ${multiarch}. You may want to override C2GOTO_INCLUDE_DIR")
mangle_clib(mangled_files ${inputs})
add_library(clibs ${mangled_files} cprover_library.cpp)
target_include_directories(clibs
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_link_libraries(clibs fmt::fmt)
