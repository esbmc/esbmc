noinst_LIBRARIES = libllvmfrontend.a
libllvmfrontend_a_SOURCES = \
      llvm_language.cpp llvm_parser.cpp parse.cpp llvm_typecheck.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util -Wno-unused-private-field
