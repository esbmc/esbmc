bin_PROGRAMS = esbmc
esbmc_SOURCES = main.cpp parseoptions.cpp bmc.cpp  \
                document_subgoals.cpp  show_vcc.cpp options.cpp
ESBMC_SOLVER_LIBS = @ESBMC_SOLVER_LIBS@
esbmc_LDADD = ../cpp/libcxx.la ../goto-symex/libsymex.a ../langapi/liblangapi.a ../ansi-c/libansic.la ../ansi-c/libclib.la ../ansi-c/cpp/libcpp.a ../ansi-c/headers/libheaders.a ../goto-programs/libgotoprograms.a ../pointer-analysis/libpointeranalysis.a ../util/libutil.a ../big-int/libbigint.a ../solvers/libsolvers.la buildidstring.o $(ESBMC_SOLVER_LIBS)
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util
AM_LDFLAGS = @ESBMC_LDFLAGS@

buildidstring.o:
	$(top_srcdir)/scripts/buildidobj.sh .
	$(top_srcdir)/ansi-c/headers/flail.sh buildidstring .//buildidobj.txt | $(CC) $(CFLAGS) -x c - -c -o ./buildidstring.o
