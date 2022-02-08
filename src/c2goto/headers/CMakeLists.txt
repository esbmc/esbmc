set(inputs limits.h pthread.h stdarg.h stdbool.h stddef.h
           # Glibc
           bits/pthreadtypes.h bits/thread-shared-types.h
           # MacOSX10.x
           sys/_types/_pthread_attr_t.h
           # MacOSX11.0 and above
           sys/_pthread/_pthread_attr_t.h
           sys/_pthread/_pthread_cond_t.h
           sys/_pthread/_pthread_condattr_t.h
           sys/_pthread/_pthread_key_t.h
           sys/_pthread/_pthread_mutex_t.h
           sys/_pthread/_pthread_mutexattr_t.h
           sys/_pthread/_pthread_once_t.h
           sys/_pthread/_pthread_rwlock_t.h
           sys/_pthread/_pthread_rwlockattr_t.h
           sys/_pthread/_pthread_t.h
           sys/_pthread/_pthread_types.h
   )
mangle(mangled_files ${CMAKE_CURRENT_SOURCE_DIR} ${inputs}
       PREFIX esbmc_libc_header_
       MACRO ESBMC_FLAIL
       ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libc_hdr.h)
add_library(c2gotoheaders ${mangled_files})
set_target_properties(c2gotoheaders PROPERTIES LINKER_LANGUAGE C
                                               PRIVATE_HEADER ${CMAKE_CURRENT_BINARY_DIR}/libc_hdr.h)
