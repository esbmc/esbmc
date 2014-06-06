bin_PROGRAMS = esbmc
esbmc_SOURCES = main.cpp parseoptions.cpp bmc.cpp  \
                document_subgoals.cpp  show_vcc.cpp options.cpp \
                kinduction_parallel.cpp
esbmc_LDADD = ../cpp/libcxx.a ../goto-symex/libsymex.a ../langapi/liblangapi.a ../ansi-c/libansic.a ../ansi-c/libclib.a ../ansi-c/cpp/libcpp.a ../ansi-c/headers/libheaders.a ../goto-programs/libgotoprograms.a ../pointer-analysis/libpointeranalysis.a ../util/libutil.a ../big-int/libbigint.a buildidstring.o
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util

buildidstring.o:
	$(top_srcdir)/scripts/buildidobj.sh .
	$(top_srcdir)/ansi-c/headers/flail.sh buildidstring .//buildidobj.txt | $(CC) $(CFLAGS) -x c - -c -o ./buildidstring.o
