set (inputs limits.h pthread.h bits/pthreadtypes.h stdarg.h stdbool.h stddef.h)
mangle(mangled_files ${CMAKE_CURRENT_SOURCE_DIR} ${inputs}
       PREFIX esbmc_libc_header_
       MACRO ESBMC_FLAIL
       ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libc_hdr.h)
add_library(c2gotoheaders ${mangled_files})
set_target_properties(c2gotoheaders PROPERTIES LINKER_LANGUAGE C
                                               PRIVATE_HEADER ${CMAKE_CURRENT_BINARY_DIR}/libc_hdr.h)
