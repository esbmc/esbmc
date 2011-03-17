DIRS = ansi-c big-int esbmc csp hoare \
       infrules intrep solvers pvs separate smvlang util \
       langapi cpp pascal symex \
       satqe satmc explain specc \
       goto-programs bplang simplifylang vcegar vsynth \
       cvclang pointer-analysis \
       goto-symex ipp prover trans bv_refinement mcp \
       smtlang php cemc interpolation scoot floatbv

all: esbmc.dir \

include config.inc
include common

###############################################################################

util.dir: big-int.dir

languages: util.dir langapi.dir \
           ansi-c.dir intrep.dir cvclang.dir smvlang.dir \
           bplang.dir

esbmc.dir: languages solvers.dir goto-symex.dir \
          pointer-analysis.dir goto-programs.dir goto-symex.dir

cemc.dir: esbmc.dir

scoot.dir: languages

explain.dir: esbmc.dir

vcegar.dir: languages satqe.dir solvers.dir

vsynth.dir: langapi.dir util.dir 

satmc.dir: solvers.dir smvlang.dir util.dir languages \
           intrep.dir trans.dir satqe.dir

ifdef MODULE_INTERPOLATION
interpolation.dir: solvers.dir langapi.dir util.dir
endif

ifdef MODULE_BV_REFINEMENT
esbmc.dir: bv_refinement.dir
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

$(patsubst %, %.dir, $(DIRS)):
	## Entering $(basename $@)
	$(MAKE) $(MAKEARGS) -C $(basename $@)

clean: $(patsubst %, %_clean, $(DIRS))
	rm -f *.o
	find . -name *.o | xargs rm -f
	find . -name *.d | xargs rm -f

dep: $(patsubst %, %_dep, $(DIRS))

$(patsubst %, %_clean, $(DIRS)):
	if [ -e $(patsubst %_clean, %, $@)/. ] ; then \
		$(MAKE) $(MAKEARGS) -C $(patsubst %_clean, %, $@) clean ; \
	fi

$(patsubst %, %_dep, $(DIRS)):
	if [ -e $(patsubst %_dep, %, $@)/. ] ; then \
		$(MAKE) $(MAKEARGS) -C $(patsubst %_dep, %, $@) dep ; \
	fi
