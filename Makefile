ESBMCDIR = $(shell pwd)
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc

all: esbmc

###############################################################################

util: big-int

infrastructure: util langapi solvers goto-symex pointer-analysis \
		goto-programs

languages: ansi-c cpp

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
		env CLEANRULE=1 $(MAKE) -C $$dir clean; \
	done
	-rm $(OBJDIR)/.depends

.PHONY: $(DIRS) clean

esbmc: $(OBJDIR)/.depends infrastructure languages

###############################################################################

include $(ESBMCDIR)/common
