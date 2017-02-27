# What we're building
EXTRA_PROGRAMS = esbmc
bin_PROGRAMS = @ESBMC_BINARY@
EXTRA_LTLIBRARIES = libesbmc.la
lib_LTLIBRARIES = @ESBMC_LIB@
noinst_LTLIBRARIES = libmain.la

# Work out what gets linked in.
ESBMC_SOLVER_LIBS = @ESBMC_SOLVER_LIBS@
libs_to_link_in = ../cpp/libcxx.la ../goto-symex/libsymex.la \
      ../langapi/liblangapi.la ../ansi-c/libansic.la ../c2goto/libclib.la \
      ../ansi-c/cpp/libcpp.la ../c2goto/headers/libheaders.la \
      ../goto-programs/libgotoprograms.la @ESBMC_CLANG_ARCHIVES@ \
      ../pointer-analysis/libpointeranalysis.la ../util/libutil.la \
      ../big-int/libbigint.la ../solvers/libsolvers.la

# An aside about python: linking it in is sketchy, but supported by libtool
# extra sources and dependencies. The real issue is that the symex convenience
# library depends on init_module_esbmc, but that depends on symex symbols,
# which is circular. By sticking python.lo at the end of each link command line
# we ensure that init_module_esbmc is pulled in by symex, but we rely on
# various symex library symbols being pulled in by the rest of esbmc, which
# isn't ideal.

# Libmain: a library of all the functional stuff in the esbmc directory,
# aside from functions like main and the build id string.
libmain_la_SOURCES = parseoptions.cpp bmc.cpp globals.cpp \
                document_subgoals.cpp  show_vcc.cpp options.cpp
EXTRA_libmain_la_SOURCES = python.cpp

# libesbmc -> shared object of all of ESBMCs Stuff (TM)
# -rpath needed otherwise libtool bails on making it shared.
libesbmc_la_LDFLAGS = -module @LIBESBMC_LDFLAGS@ -rpath ${libdir} -release @ESBMC_VERSION_STR@
libesbmc_la_LIBADD = $(libs_to_link_in) $(ESBMC_SOLVER_LIBS) ./libmain.la @ESBMC_PYTHON_OBJ@
libesbmc_la_SOURCES = ltbuildidstring.c
EXTRA_libesbmc_la_DEPENDENCIES = @ESBMC_PYTHON_OBJ@

# ESBMC binary: fairly self explanatory
esbmc_LDADD = ./libmain.la $(libs_to_link_in) $(ESBMC_SOLVER_LIBS) @ESBMC_PYTHON_OBJ@ @ESBMC_LDFLAGS@
esbmc_SOURCES = main.cpp buildidstring.c
EXTRA_esbmc_DEPENDENCIES = @ESBMC_PYTHON_OBJ@

AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util

buildidstring.c: $(libs_to_link_in) $(esbmc_SOURCES) $(libmain_la_SOURCES) python.cpp
	$(top_srcdir)/scripts/buildidobj.sh .
	$(top_srcdir)/c2goto/headers/flail.sh buildidstring .//buildidobj.txt > $@

# Second instance of buildidstring to appease libtool
ltbuildidstring.c: $(libs_to_link_in) $(esbmc_SOURCES) $(libmain_la_SOURCES) python.cpp
	$(top_srcdir)/c2goto/headers/flail.sh buildidstring .//buildidobj.txt > $@

esbmcincludedir = $(includedir)/esbmc
esbmcinclude_HEADERS = bmc.h document_subgoals.h parseoptions.h version.h

