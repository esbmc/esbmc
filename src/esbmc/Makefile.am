bin_PROGRAMS = esbmc
esbmc_SOURCES = main.cpp parseoptions.cpp bmc.cpp  \
                document_subgoals.cpp show_vcc.cpp options.cpp
ESBMC_SOLVER_LIBS = @ESBMC_SOLVER_LIBS@
libs_to_link_in = ../cpp/libcxx.la ../goto-symex/libsymex.a ../langapi/liblangapi.a \
                  ../ansi-c/libansic.la ../c2goto/libclib.la ../ansi-c/cpp/libcpp.a \
                  ../c2goto/headers/libheaders.a ../goto-programs/libgotoprograms.a \
                  @ESBMC_CLANG_ARCHIVES@ \
                  ../pointer-analysis/libpointeranalysis.a ../util/libutil.a \
                  ../big-int/libbigint.a ../solvers/libsolvers.la buildidstring.o
esbmc_LDADD = $(libs_to_link_in) $(ESBMC_SOLVER_LIBS)
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util
AM_LDFLAGS = @ESBMC_LDFLAGS@

buildidstring.o: $(libs_to_link_in) $(esbmc_SOURCES)
	$(top_srcdir)/scripts/buildidobj.sh .
	$(top_srcdir)/c2goto/headers/flail.sh buildidstring .//buildidobj.txt | $(CC) $(CFLAGS) -x c - -c -o ./buildidstring.o
