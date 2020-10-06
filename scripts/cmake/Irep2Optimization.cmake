# Module to configure Irep2 precompiled headers
# This is required by MSVC

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	message(AUTHOR_WARNING "GCC-9 throws a Segmentation fault. We should report this or try newer versions")
	return()
endif()

message(WARNING ${CMAKE_CXX_COMPILER_ID})
target_precompile_headers(util_esbmc PRIVATE ${CMAKE_SOURCE_DIR}/src/util/irep2.h ${CMAKE_SOURCE_DIR}/src/util/irep2_expr.h)
target_precompile_headers(clangcfrontend_stuff REUSE_FROM util_esbmc)
target_precompile_headers(clangcppfrontend REUSE_FROM util_esbmc)
target_precompile_headers(gotoprograms REUSE_FROM util_esbmc)
target_precompile_headers(symex REUSE_FROM util_esbmc)
target_precompile_headers(pointeranalysis REUSE_FROM util_esbmc)