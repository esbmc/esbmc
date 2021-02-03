# Module to utilities used throughout the project


set(ESBMC_BIN "${CMAKE_BINARY_DIR}/src/esbmc/esbmc")
if(WIN32)
    set(ESBMC_BIN "${CMAKE_INSTALL_PREFIX}/bin/esbmc.exe")
elseif(APPLE)
    set(MACOS_ESBMC_WRAPPER "#!/bin/sh\n${ESBMC_BIN} -I${C2GOTO_INCLUDE_DIR} $@")
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