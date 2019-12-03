add_subdirectory (headers)

if(DEFINED C2GOTO_INCLUDE_DIR)
  set(multiarch "${C2GOTO_INCLUDE_DIR}")
else()
  set(multiarch "/usr/include/${CMAKE_LIBRARY_ARCHITECTURE}")
endif()

set (clib32_config clib32.c "")
set (clib64_config clib64.c "--64")
set (clib32f_config clib32_fp.c "--floatbv")
set (clib64f_config clib64_fp.c "--64;--floatbv")

set (inputs clib32_config clib64_config clib32f_config clib64f_config)

function(mangle_clib output)
  set(result)
  foreach(in_f ${ARGN})
    set (in_f "${${in_f}}")
    list(GET in_f 0 in_file)
    list(GET in_f 1 in_flags)
    file(RELATIVE_PATH out_file ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${in_file})
    string(REGEX REPLACE .c .goto out_goto "${in_file}")
    string(REGEX REPLACE .c "" barename "${in_file}")
    set(out_goto "${CMAKE_CURRENT_BINARY_DIR}/${out_goto}")
    set(out_file "${CMAKE_CURRENT_BINARY_DIR}/${out_file}")

    add_custom_command(OUTPUT ${out_goto}
      COMMAND sh -c "${CMAKE_CURRENT_BINARY_DIR}/c2goto -I ${multiarch} ${CMAKE_CURRENT_SOURCE_DIR}/library/*.c ${LIBM_PATTERN} ${in_flags} --output ${out_goto}"
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/c2goto
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating libc model ${out_goto}"
      VERBATIM
      )

    add_custom_command(OUTPUT ${out_file}
      COMMAND ${CMAKE_SOURCE_DIR}/src/c2goto/headers/flail.sh ${barename} ${out_goto} > ${out_file}
      DEPENDS ${out_goto}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Converting libc models ${out_file} to data"
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
add_library(cprover_blank_library cprover_blank_library.cpp)
target_include_directories(cprover_blank_library 
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${Boost_INCLUDE_DIRS} 
)

add_executable(c2goto c2goto.cpp)
target_include_directories(c2goto
    PRIVATE ${Boost_INCLUDE_DIRS} 
)
target_link_libraries(c2goto ansicfrontend clangcfrontend cppfrontend gotoprograms langapi util_esbmc bigint c2gotoheaders cprover_blank_library "${Boost_LIBRARIES}")

