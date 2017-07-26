noinst_LTLIBRARIES = libclangcppfrontend.la
libclangcppfrontend_la_SOURCES = \
      clang_cpp_language.cpp clang_cpp_convert.cpp clang_cpp_adjust_expr.cpp \
      expr2cpp.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS_WARNINGS) -I$(top_srcdir) -Wno-unused-private-field \
      -Wno-unused-parameter -Wno-strict-aliasing

clangcppfrontendincludedir = $(includedir)/clang-cpp-frontend
clangcppfrontendinclude_HEADERS = clang_cpp_language.h clang_cpp_adjust.h \
      clang_cpp_convert.h expr2cpp.h
