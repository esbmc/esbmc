ACLOCAL_AMFLAGS = -I scripts/build-aux/m4

SUBDIRS = big-int util langapi @ESBMC_CLANG_SUBDIRS@ ansi-c cpp goto-programs \
          c2goto solvers goto-symex pointer-analysis esbmc

licensedir = $(prefix)/license
license_DATA = scripts/licenses/BOOLECTOR_LICENSE.txt scripts/licenses/Z3_LICENSE.txt COPYING

maindir = $(prefix)
main_DATA = scripts/README scripts/release-notes.txt
