all: esbmc

ifndef SATDIR
SATDIR= $(shell pwd)
endif

export SATDIR

include config.inc
include local.inc
include common

###############################################################################

util: big-int

infrastructure: util langapi solvers goto-symex pointer-analysis \
		goto-programs goto-symex

languages: ansi-c cpp

# Ansi-c builds its library using infrastructure facilities.
ansi-c: infrastructure

###############################################################################

DIRS= big-int util langapi solvers goto-symex pointer-analysis goto-programs \
      goto-symex ansi-c esbmc cpp

$(DIRS):
	$(MAKE) -C $@

clean:
	for dir in $(DIRS); do \
		$(MAKE) -C $$dir clean; \
	done

.PHONY: $(DIRS) clean

esbmc: infrastructure languages

###############################################################################

ifdef MODULE_INTERPOLATION
interpolation.dir: solvers.dir langapi.dir util.dir
endif

ifdef MODULE_SATQE
all: satmc.dir
endif

ifdef MODULE_SMTLANG
languages: smtlang.dir
endif

ifdef MODULE_CPP
languages: cpp
endif

ifdef MODULE_PHP
languages: php.dir
endif

ifdef MODULE_CSP
languages: csp.dir
endif

ifdef MODULE_PVS
languages: pvs.dir
endif

ifdef MODULE_SPECC
languages: specc.dir
endif

ifdef MODULE_PASCAL
languages: pascal.dir
endif

ifdef MODULE_SIMPLIFYLANG
languages: simplifylang.dir
endif

ifdef MODULE_CEMC
all: cemc.dir
endif

ifdef MODULE_FLOATBV
solvers.dir: floatbv.dir
endif
