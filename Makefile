ESBMCDIR = $(shell pwd)
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc

all: esbmc

###############################################################################

util: big-int

infrastructure: util langapi solvers goto-symex pointer-analysis \
		goto-programs

languages: ansi-c

# Ansi-c builds its library using infrastructure facilities.
ansi-c: infrastructure

###############################################################################

DIRS= big-int util langapi solvers goto-symex pointer-analysis goto-programs \
      ansi-c esbmc cpp

NJOBS := $(shell if [ -f /proc/cpuinfo ]; \
                    then echo `cat /proc/cpuinfo | grep 'processor' | wc -l`; \
                    else echo 1; fi)
$(DIRS):
	$(MAKE) -j$(NJOBS) -C $@

clean:
	for dir in $(DIRS); do \
		$(MAKE) -C $$dir clean; \
	done
	rm $(OBJDIR)/.depends

.PHONY: $(DIRS) clean

esbmc: $(OBJDIR)/.depends infrastructure languages

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

include $(ESBMCDIR)/common
