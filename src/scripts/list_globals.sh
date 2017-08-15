#!/bin/sh

if test ! -e esbmc; then
  echo "Please run from src/ dir";
  exit 1
fi

echo "Printed below are a set of global and/or static variables in the ESBMC binary. It may be incomplete, incorrect, out of date, irritating, harmful to cats, and all of the above."

echo "Proper globals, minus known benign fields:"

# This: Fetches a list of symbols from esbmc, that are global data objects.
# Convert back from C++ name manging and print. Strip out the following:
# irept::*, which contains global name irep_idt's. Not interesting at all.
# exprt::*, same as the above.
# typet::*, same as the above.
# *::field_names, just names of irep2 fields
objdump -t $1 | awk '{if ($2 == "g" && $3 == "O") print $6;}' | c++filt | grep -v '^irept::.*$' | grep -v '^exprt::.*$' | grep -v '^typet::.*$' | grep -v '.*::field_names$' | grep -v '^clang::' | grep -v '^llvm::' | grep -v '^msat::' | sort -u

echo "Local objects with data storage:"

# And this: fetches a list of local symbols that are data objects, and in either
# .data or .bss, thus editable runtime data. Strip out the following:
# std::__ioinit, not sure what it is, but not esbmc related.
objdump -t $1 | awk '{if ($2 == "l" && $3 == "O" && ($4 == ".bss" || $4 == ".data")) print $6;}' | c++filt | grep -v 'std::__ioinit' |  grep -v '^clang::' | grep -v '^llvm::' | grep -v '^msat::'  | sort -u
