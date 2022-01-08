set (inputs limits.hs pthread.hs pthreadtypes.hs stdarg.hs stdbool.hs stddef.hs)
mangle(mangled_files ${CMAKE_CURRENT_SOURCE_DIR} ${inputs} PREFIX esbmc_libc_header_)
add_library(c2gotoheaders ${mangled_files})
set_target_properties(c2gotoheaders PROPERTIES LINKER_LANGUAGE C)
