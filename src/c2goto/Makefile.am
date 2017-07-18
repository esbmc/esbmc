SUBDIRS = headers

noinst_LTLIBRARIES = libclib.la

noinst_PROGRAMS = c2goto
c2goto_SOURCES = c2goto.cpp cprover_blank_library.cpp
c2goto_LDADD = ../clang-c-frontend/libclangcfrontend.la \
      ../clang-c-frontend/headers/libclangheaders.la \
      ../clang-c-frontend/AST/libclangcfrontendast.la \
      ../ansi-c/libansic.la ../ansi-c/libansicscanner.la \
      ../ansi-c/cpp/libcpp.la ../goto-programs/libgotoprograms.la \
      ../langapi/liblangapi.la ../util/libutil.la \
      ../big-int/libbigint.la headers/libheaders.la \
      @ESBMC_LDFLAGS@
c2goto_DEPENDENCIES = ../clang-c-frontend/libclangcfrontend.la \
      ../clang-c-frontend/headers/libclangheaders.la \
      ../clang-c-frontend/AST/libclangcfrontendast.la
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir)

libclib_la_SOURCES = cprover_library.cpp clib32.c clib64.c clib32_fp.c \
      clib64_fp.c

clib32.c: c2goto $(srcdir)/library/*.c $(srcdir)/library/libm/*.c
	./c2goto $(srcdir)/library/*.c $(srcdir)/library/libm/*.c --output clib32.goto
	$(srcdir)/headers/flail.sh clib32 clib32.goto > $@

clib64.c: c2goto $(srcdir)/library/*.c $(srcdir)/library/libm/*.c
	./c2goto $(srcdir)/library/*.c $(srcdir)/library/libm/*.c --64 --output clib64.goto
	$(srcdir)/headers/flail.sh clib64 clib64.goto > $@

clib32_fp.c: c2goto $(srcdir)/library/*.c $(srcdir)/library/libm/*.c
	./c2goto $(srcdir)/library/*.c $(srcdir)/library/libm/*.c --floatbv --output clib32_fp.goto
	$(srcdir)/headers/flail.sh clib32_fp clib32_fp.goto > $@

clib64_fp.c: c2goto $(srcdir)/library/*.c $(srcdir)/library/libm/*.c
	./c2goto $(srcdir)/library/*.c $(srcdir)/library/libm/*.c --64 --floatbv --output clib64_fp.goto
	$(srcdir)/headers/flail.sh clib64_fp clib64_fp.goto > $@

clean-local:
	-rm clib64.goto clib32.goto clib64.c clib32.c
.PHONY: clean-local

.NOTPARALLEL: all
