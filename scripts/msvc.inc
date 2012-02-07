Z31 = $(SATDIR)/z3_win
Z3 = $(shell cygpath -w $(Z31) | sed s/\\\\/\\\\\\\\/g )

WINSDKDIR=$(shell reg query HKLM\\SOFTWARE\\Microsoft\\Microsoft\ SDKs\\Windows /v CurrentInstallFolder | grep REG_SZ | sed "s\# *CurrentInstallFolder *REG_SZ *\(.*\)\#\1\#" | sed s/\\\\/\\\\\\\\/g)

VCDIR=$(shell reg query HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\VisualStudio\\10.0 /v ShellFolder | grep REG_SZ | sed "s\# *ShellFolder *REG_SZ *\(.*\)\#\1\#" | sed s/\\\\/\\\\\\\\/g)VC

CYGVCDIR="$(shell cygpath -u "$(VCDIR)")"
VCINCLUDES+=/I"$(VCDIR)\\include" /I"$(WINSDKDIR)\\include"
VCBINS=$(CYGVCDIR)/bin/amd64
WINESBMCDIR= $(shell cygpath -w $(ESBMCDIR) | sed s/\\\\/\\\\\\\\/g )
LD=$(VCBINS)\\link.exe
C2GOTO_INCS=-I"$(VCDIR)\\include" -I"$(WINSDKDIR)\\include" -D_MSVC
CXX=$(VCBINS)\\cl.exe
CC=$(VCBINS)\\cl.exe
FINLINKBIN=$(CYGVCDIR)/bin/amd64/cl.exe
FINLINKOUT= /Fe
BLOBGENLD=x86_64-w64-mingw32-ld
INCCMD=/I
PATHSEP=\\
CXXFLAGS+=/I"$(VCDIR)\\include"
PLATLIBS="C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\lib\\x64\\wsock32.lib"
FINLINKTAIL= $(CXXFLAGS) /link /LIBPATH:"$(VCDIR)\\lib\\amd64" /LIBPATH:"C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\lib\\x64\\" $(PLATLIBS) $(LIBS)
PARTLINKLD=$(VCBINS)/lib.exe
PARTLINKOUT= /OUT:
PARTLINKOPTS= 

OPTS ?= /O2
