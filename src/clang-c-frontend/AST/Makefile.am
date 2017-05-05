noinst_LTLIBRARIES = libclangcfrontendast.la

libclangcfrontendast_la_SOURCES = build_ast.cpp

# Can't rely on clang headers not warning, so disable werror
AM_CXXFLAGS = $(ESBMC_CXXFLAGS_WARNINGS) -I$(top_srcdir) -I$(top_srcdir)/util \
      -Wno-unused-private-field -Wno-unused-parameter -Wno-strict-aliasing \
      -fno-rtti

clangcfrontendastincludedir = $(includedir)/clang-c-frontend/AST
clangcfrontendastinclude_HEADERS = build_ast.h
