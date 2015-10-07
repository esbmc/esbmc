noinst_LIBRARIES = libllvmfrontend.a
libllvmfrontend_a_SOURCES = \
      llvm_language.cpp llvm_convert.cpp llvm_main.cpp llvm_adjust_expr.cpp typecast.cpp \
      llvm_adjust_code.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util -Wno-unused-private-field
