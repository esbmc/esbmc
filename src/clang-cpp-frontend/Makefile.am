noinst_LTLIBRARIES = libclangcppfrontend.la
libclangcppfrontend_la_SOURCES = \
      clang_cpp_language.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS_WARNINGS) -I$(top_srcdir) -I$(top_srcdir)/util \
      -Wno-unused-private-field

clangcppfrontendincludedir = $(includedir)/clang-cpp-frontend
clangcppfrontendinclude_HEADERS = clang_cpp_language.h
