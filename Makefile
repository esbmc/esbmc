DIRS = big-int esbmc hoare infrules intrep solvers separate smvlang \
	util langapi cpp symex satqe goto-programs bplang cvclang \
	pointer-analysis goto-symex trans smtlang ansi-c

all: esbmc

include config.inc
include local.inc
include common

###############################################################################

$(DIRS):
	$(MAKE) -C $@

.PHONY: $(DIRS) infrastructure langauges

util: big-int

infrastructure: util langapi solvers goto-symex pointer-analysis \
		goto-programs goto-symex

languages: intrep ansi-c

# Ansi-c builds its library using infrastructure facilities.
ansi-c: infrastructure

esbmc: infrastructure languages

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
languages: cpp.dir
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

.PHONY: clean

clean:
	for dir in $(DIRS); do \
		$(MAKE) -C $$dir clean; \
	done
