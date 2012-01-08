# If SATDIR isn't set yet, assume the user has solvers in their home dir.
SATDIR ?= $(HOME)

# Hack around MSVC linking against Z3.
ifdef WIN_MSVC
Z31 = $(SATDIR)/z3_win
Z3 = $(shell cygpath -w $(Z31) | sed s/\\\\/\\\\\\\\/g )
endif

ifdef WIN_MINGW32
Z3 = $(SATDIR)/z3_win
endif

Z3GCC = $(SATDIR)/z3_win

# Only use these solvers on linux
ifndef WIN_MSVC
ifndef WIN_MINGW32
BOOLECTOR = $(SATDIR)/boolector
MINISAT = $(SATDIR)/MiniSat-p_v1.14
endif
endif
