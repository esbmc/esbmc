# Module to utilities used throughout the project


set(ESBMC_BIN "${CMAKE_BINARY_DIR}/src/esbmc/esbmc")
if(WIN32)
    set(ESBMC_BIN "${CMAKE_INSTALL_PREFIX}/bin/esbmc.exe")
elseif(APPLE)
    set(MACOS_ESBMC_WRAPPER "#!/bin/sh\n${ESBMC_BIN} --sysroot ${C2GOTO_SYSROOT} $@")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/macos-wrapper.sh ${MACOS_ESBMC_WRAPPER})
    set(ESBMC_BIN "${CMAKE_CURRENT_BINARY_DIR}/macos-wrapper.sh")
endif()

# Assert that a variable is defined
function(assert_variable_is_defined VAR)
    if(DEFINED ${VAR})
        message(STATUS "${VAR}: ${${VAR}}")
    else()
        message(FATAL_ERROR "${VAR} must be defined")
    endif()
endfunction()

# Get all subdirectories of a folder
# https://stackoverflow.com/a/7788165/7483927
macro(SUBDIRLIST result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

function(mangle output_c dir)
  # optional flags
  set(kw0 SINGLE           # accumulate all outputs in the single .c file ${output_c}
     )
  # optional 1-parameter keywords to this function
  set(kw1 MACRO            # sets flail.py --macro parameter to the given argument
          ACC_HEADERS_INTO # generates a header that #include-s all generated headers
          WILDCARD         # globs ${dir}/${MANGLE_WILDCARD} instead of using the unparsed arguments as filenames under ${dir}
          PREFIX           # sets flail.py --prefix parameter to the given argument
     )
  cmake_parse_arguments(MANGLE "${kw0}" "${kw1}" "" ${ARGN})

  if (MANGLE_WILDCARD)
    set(single_file_desc "${MANGLE_WILDCARD} in ${dir}/")
    file(GLOB inputs RELATIVE ${dir} CONFIGURE_DEPENDS ${dir}/${MANGLE_WILDCARD} ${MANGLE_UNPARSED_ARGUMENTS})
  else()
    set(inputs ${MANGLE_UNPARSED_ARGUMENTS})
    list(JOIN ", " inputs single_file_desc)
    set(single_file_desc "${inputs} in ${dir}")
  endif()

  set(cmd0 ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py)
  if (MANGLE_MACRO)
    list(APPEND cmd0 --macro ${MANGLE_MACRO})
  endif()

  if (MANGLE_PREFIX)
    list(APPEND cmd0 --prefix ${MANGLE_PREFIX})
  endif()

  if (MANGLE_SINGLE)
    set(outputs ${output_c})
    list(TRANSFORM inputs PREPEND ${dir}/ OUTPUT_VARIABLE dep)
    list(APPEND dep ${CMAKE_SOURCE_DIR}/scripts/flail.py)
    set(cmd ${cmd0} -o ${output_c})
    if (MANGLE_ACC_HEADERS_INTO)
      list(APPEND outputs ${MANGLE_ACC_HEADERS_INTO})
      list(APPEND cmd --header ${MANGLE_ACC_HEADERS_INTO})
    endif()
    add_custom_command(OUTPUT ${outputs}
                       COMMAND ${cmd} ${inputs}
                       DEPENDS ${dep}
                       WORKING_DIRECTORY ${dir}
                       COMMENT "Converting ${single_file_desc} to data"
                       VERBATIM)
  else()
    set(result_c "")
    set(includes "")
    foreach(in_f ${inputs})
      set(out "${CMAKE_CURRENT_BINARY_DIR}/${in_f}")
      set(outputs ${out}.c)
      set(cmd ${cmd0} -o ${out}.c)
      set(dep ${dir}/${in_f} ${CMAKE_SOURCE_DIR}/scripts/flail.py)
      if (MANGLE_ACC_HEADERS_INTO)
        list(APPEND outputs ${out}.h)
        list(APPEND cmd --header ${out}.h)
        list(APPEND dep ${MANGLE_ACC_HEADERS_INTO})
        list(APPEND includes "#include \"${out}.h\"")
      endif()
      add_custom_command(OUTPUT ${outputs}
        COMMAND ${cmd} ${in_f}
        DEPENDS ${dep}
        WORKING_DIRECTORY ${dir}
        COMMENT "Converting ${in_f} to data"
        VERBATIM
        )
      list(APPEND result_c ${out}.c)
    endforeach()
    set(${output_c} "${result_c}" PARENT_SCOPE)
    if (MANGLE_ACC_HEADERS_INTO)
      list(JOIN includes "\n" includes)
      file(WRITE ${MANGLE_ACC_HEADERS_INTO} "${includes}")
    endif()
  endif()
endfunction()

function(download_zip_and_extract ID URL)
  # TODO: might be a good idea to add sha1 checks
  if(NOT EXISTS ${CMAKE_BINARY_DIR}/${ID}.zip)
    message(STATUS "Downloading ${ID} from ${URL}")
    file(DOWNLOAD ${URL} ${CMAKE_BINARY_DIR}/${ID}.zip SHOW_PROGRESS)
  endif()

  if(NOT EXISTS ${CMAKE_BINARY_DIR}/${ID})
    message(STATUS "Extracting ${ID}") 
    file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/${ID}.zip DESTINATION ${CMAKE_BINARY_DIR}/${ID})
  endif()

endfunction()