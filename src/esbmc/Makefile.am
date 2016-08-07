EXTRA_PROGRAMS = esbmc
bin_PROGRAMS = @ESBMC_BINARY@
EXTRA_LTLIBRARIES = libesbmc.la
lib_LTLIBRARIES = libmain.la @ESBMC_SHARED_LIB@
ESBMC_SOLVER_LIBS = @ESBMC_SOLVER_LIBS@
libs_to_link_in = ../cpp/libcxx.la ../goto-symex/libsymex.la ../langapi/liblangapi.la \
                  ../ansi-c/libansic.la ../ansi-c/libclib.la ../ansi-c/cpp/libcpp.la \
                  ../ansi-c/headers/libheaders.la ../goto-programs/libgotoprograms.la \
                  @ESBMC_CLANG_ARCHIVES@ \
                  ../pointer-analysis/libpointeranalysis.la ../util/libutil.la \
                  ../big-int/libbigint.la ../solvers/libsolvers.la
libesbmc_la_LIBADD = $(libs_to_link_in) $(ESBMC_SOLVER_LIBS) ./libmain.la
libesbmc_la_SOURCES = ltbuildidstring.c
libmain_la_SOURCES = parseoptions.cpp bmc.cpp \
                document_subgoals.cpp  show_vcc.cpp options.cpp
EXTRA_libmain_la_SOURCES = python.cpp
# python.lo will *not* be built unless it's specified in the dependencies
EXTRA_libmain_la_DEPENDENCIES = @ESBMC_PYTHON_OBJ@
libmain_la_LIBADD = @ESBMC_PYTHON_OBJ@
esbmc_SOURCES = main.cpp buildidstring.c
libesbmc_la_LDFLAGS = -module -shared -rpath ${libdir} # Makes this shared
esbmc_LDADD = ./libmain.la $(libs_to_link_in) $(ESBMC_SOLVER_LIBS)
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util
AM_LDFLAGS = @ESBMC_LDFLAGS@

buildidstring.c: $(libs_to_link_in) $(esbmc_SOURCES) $(libmain_la_SOURCES)
	$(top_srcdir)/scripts/buildidobj.sh .
	$(top_srcdir)/ansi-c/headers/flail.sh buildidstring .//buildidobj.txt > $@

# Second instance of buildidstring to appease libtool
ltbuildidstring.c: $(libs_to_link_in) $(esbmc_SOURCES) $(libmain_la_SOURCES)
	$(top_srcdir)/ansi-c/headers/flail.sh buildidstring .//buildidobj.txt > $@
