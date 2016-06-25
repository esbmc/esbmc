bin_PROGRAMS = esbmc
EXTRA_LTLIBRARIES = libesbmc.la
lib_LTLIBRARIES = libmain.la @ESBMC_SHARED_LIB@
ESBMC_SOLVER_LIBS = @ESBMC_SOLVER_LIBS@
libs_to_link_in = ../cpp/libcxx.la ../goto-symex/libsymex.a ../langapi/liblangapi.a \
                  ../ansi-c/libansic.la ../ansi-c/libclib.la ../ansi-c/cpp/libcpp.a \
                  ../clang-c-frontend/libclangcfrontend.a ../ansi-c/headers/libheaders.a \
                  ../clang-cpp-frontend/libclangcppfrontend.a ../goto-programs/libgotoprograms.a \
                  ../pointer-analysis/libpointeranalysis.a ../util/libutil.a \
                  ../big-int/libbigint.a ../solvers/libsolvers.la buildidstring.o
libesbmc_la_SOURCES = $(libs_to_link_in) parseoptions.cpp bmc.cpp  \
                document_subgoals.cpp  show_vcc.cpp options.cpp
libmain_la_SOURCES = parseoptions.cpp bmc.cpp \
                document_subgoals.cpp  show_vcc.cpp options.cpp
esbmc_SOURCES = main.cpp
libesbmc_la_LDFLAGS = -module # Makes this shared
esbmc_LDADD = ./libmain.la $(libs_to_link_in) $(ESBMC_SOLVER_LIBS)
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util
AM_LDFLAGS = @ESBMC_LDFLAGS@

buildidstring.o: $(libs_to_link_in) $(esbmc_SOURCES)
	$(top_srcdir)/scripts/buildidobj.sh .
	$(top_srcdir)/ansi-c/headers/flail.sh buildidstring .//buildidobj.txt | $(CC) $(CFLAGS) -x c - -c -o ./buildidstring.o
