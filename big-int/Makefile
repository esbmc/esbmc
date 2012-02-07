ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
all: $(OBJDIR)/.deps test-bigint

RM = rm -f
SRCS= bigint-test.cpp bigint-func.cpp bigint.cpp

include $(ESBMCDIR)/common

clean:
	-$(RM) $(OBJS)
	-rm -f $(OBJDIR)/.deps

test-bigint: $(OBJS)
	$(FINLINKBIN) $(FINLINKOUT)test-bigint $(OBJS) $(FINLINKTAIL)
