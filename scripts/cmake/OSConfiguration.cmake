# Module to configure the build based on the OS

#############################
# ABOUT
#############################

#[[
The build system have flags based on the current system, this may 
be for finding executables or custom links

The global variables that all modules NEED to define are:

- LIBGOMP_LIB: Flags to link libgomp (used by Z3)
- OS_FLEX_FLAGS: Flags used for Flex targets this may add
    defines exclusive to the OS

]]

# This function will check if a variable is defined
function(assert_variable_is_defined VAR)
    if(DEFINED ${VAR})
        message(STATUS "${VAR}: ${${VAR}}")
    else()
        message(FATAL_ERROR "${VAR} must be defined")
    endif()
endfunction()

include(AppleConfiguration)
include(UnixConfiguration)
include(WindowsConfiguration)

# Assert that the variables were assigned at all
assert_variable_is_defined(LIBGOMP_LIB)
assert_variable_is_defined(OS_FLEX_FLAGS)