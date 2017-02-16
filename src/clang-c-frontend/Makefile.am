SUBDIRS = headers

noinst_LTLIBRARIES = libclangcfrontend.la

libclangcfrontend_la_SOURCES = clang_c_language.cpp clang_c_convert.cpp \
      clang_c_main.cpp clang_c_adjust_expr.cpp typecast.cpp \
      clang_c_adjust_code.cpp clang_c_convert_literals.cpp clang_headers.cpp

# Can't rely on clang headers not warning, so disable werror
AM_CXXFLAGS = $(ESBMC_CXXFLAGS_WARNINGS) -I$(top_srcdir) -I$(top_srcdir)/util \
      -Wno-unused-private-field -Wno-unused-parameter -Wno-strict-aliasing

clangcfrontendincludedir = $(includedir)/clang-c-frontend
clangcfrontendinclude_HEADERS = clang_c_adjust.h clang_c_convert.h \
      clang_c_language.h clang_c_main.h typecast.h
