# Configure and enable -Werror

if(ENABLE_WERROR)
    message(STATUS "Compiling with Warnings are Errors")
    add_compile_options(-Werror)
    add_compile_options(-Wno-error=invalid-pch) # TODO: fix before merge
    add_compile_options(-Wno-error=unused-parameter) # TODO: fix before merge
    add_compile_options(-Wno-error=unused-but-set-variable) # TODO: fix before merge
endif()
