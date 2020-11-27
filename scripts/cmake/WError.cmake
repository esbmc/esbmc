# Configure and enable -Werror

if(ENABLE_WERROR)
    message(STATUS "Compiling with Warnings are Errors")
    add_compile_options(-Wall -Wextra -Werror)    
endif()