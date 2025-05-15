# Module to configure Irep2 precompiled headers
# This is required by MSVC

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	message(AUTHOR_WARNING "GCC-9 throws a Segmentation fault. We should report this or try newer versions")
	return()
endif()

if(${CMAKE_VERSION} VERSION_LESS "3.16.0")
    message(AUTHOR_WARNING "This optimization is available only for CMake 3.16+")
    return()
endif()

target_precompile_headers(util_esbmc PRIVATE ${CMAKE_SOURCE_DIR}/src/irep2/irep2.h ${CMAKE_SOURCE_DIR}/src/irep2/irep2_expr.h)
target_precompile_headers(clangcfrontend_stuff REUSE_FROM util_esbmc)
target_precompile_headers(clangcppfrontend REUSE_FROM util_esbmc)
target_precompile_headers(gotoprograms REUSE_FROM util_esbmc)
target_precompile_headers(symex REUSE_FROM util_esbmc)
target_precompile_headers(pointeranalysis REUSE_FROM util_esbmc)
