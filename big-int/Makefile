ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
all: test-bigint

RM = rm -f
SRCS= bigint-test.cpp bigint-func.cpp bigint.cpp

include $(ESBMCDIR)/common

clean:
	-$(RM) $(OBJS)
	-rm -f .deps

test-bigint: $(OBJS)
	$(CXX) $(CXXFLAGS) -o test-bigint $(OBJS)
