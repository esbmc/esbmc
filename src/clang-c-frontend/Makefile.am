noinst_LIBRARIES = libclangcfrontend.a
libclangcfrontend_a_SOURCES = \
      clang_c_language.cpp clang_c_convert.cpp clang_c_main.cpp clang_c_adjust_expr.cpp typecast.cpp \
      clang_c_adjust_code.cpp clang_c_convert_literals.cpp clang_headers.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util -Wno-unused-private-field
