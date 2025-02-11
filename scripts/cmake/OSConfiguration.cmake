# Module to configure the build based on the OS

#############################
# ABOUT
#############################

#[[
The build system have flags based on the current system, this may 
be for finding executables or custom links

The global variables that all modules NEED to define are:

- LIBGOMP_LIB: Flags to link libgomp (used by Z3);
- OS_FLEX_SMTLIB_FLAGS: Flags used for Flex targets. This may add defines exclusive to
    the OS (for smtlib);
- OS_X86_INCLUDE_FOLDER: Path to the default multiarch headers of the system,
    when generating goto models some of those are in 32bits. This is were they
    will find the headers by adding -I <folder>;
- OS_C2GOTO_FLAGS: Flags used by c2goto. This may add defines exclusive for the
    OS;
- OS_Z3_LIBS: Flags used by Z3 link;
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
assert_variable_is_defined(OS_FLEX_SMTLIB_FLAGS)
assert_variable_is_defined(OS_X86_INCLUDE_FOLDER)
assert_variable_is_defined(OS_C2GOTO_FLAGS)
assert_variable_is_defined(OS_INCLUDE_LIBS)
assert_variable_is_defined(OS_Z3_LIBS)