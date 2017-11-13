SUBDIRS = headers

noinst_LTLIBRARIES = libclib.la

noinst_PROGRAMS = c2goto
c2goto_SOURCES = c2goto.cpp cprover_blank_library.cpp
c2goto_LDADD = ../ansi-c/libansic.la ../ansi-c/libansicscanner.la \
      @ESBMC_CLANG_ARCHIVES@ ../big-int/libbigint.la ../cpp/libcxx.la \
      ../goto-programs/libgotoprograms.la ../langapi/liblangapi.la \
      ../util/libutil.la ../ansi-c/cpp/libcpp.la headers/libheaders.la \
       @ESBMC_LDFLAGS@
c2goto_DEPENDENCIES = @ESBMC_CLANG_ARCHIVES@
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir)

libclib_la_SOURCES = cprover_library.cpp clib32.c clib64.c clib32_fp.c \
      clib64_fp.c

clib32.c: c2goto $(srcdir)/library/*.c @ESBMC_LIBM@
	./c2goto $(srcdir)/library/*.c @ESBMC_LIBM@ --output clib32.goto
	$(srcdir)/headers/flail.sh clib32 clib32.goto > $@

clib64.c: c2goto $(srcdir)/library/*.c @ESBMC_LIBM@
	./c2goto $(srcdir)/library/*.c @ESBMC_LIBM@ --64 --output clib64.goto
	$(srcdir)/headers/flail.sh clib64 clib64.goto > $@

clib32_fp.c: c2goto $(srcdir)/library/*.c @ESBMC_LIBM@
	./c2goto $(srcdir)/library/*.c @ESBMC_LIBM@ --floatbv --output clib32_fp.goto
	$(srcdir)/headers/flail.sh clib32_fp clib32_fp.goto > $@

clib64_fp.c: c2goto $(srcdir)/library/*.c @ESBMC_LIBM@
	./c2goto $(srcdir)/library/*.c @ESBMC_LIBM@ --64 --floatbv --output clib64_fp.goto
	$(srcdir)/headers/flail.sh clib64_fp clib64_fp.goto > $@

clean-local:
	-rm clib64.goto clib32.goto clib64.c clib32.c
.PHONY: clean-local

.NOTPARALLEL: all
