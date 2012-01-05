ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
all: $(OBJDIR)/.deps test-bigint

RM = rm -f
SRCS= bigint-test.cpp bigint-func.cpp bigint.cpp

include $(ESBMCDIR)/common

clean:
	-$(RM) $(OBJS)
	-rm -f $(OBJDIR)/.deps

test-bigint: $(OBJS)
	$(CYGVCDIR)/bin/amd64/cl.exe $(CXXFLAGS) /Fetest-bigint $(OBJS) /link /LIBPATH:$(VCDIR)\\lib\\amd64 /LIBPATH:"C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\lib\\x64\\"
