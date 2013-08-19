# Don't autodetect if any of the following are already set:
ifndef LINUX
ifndef WIN_MINGW32
ifndef ONAMAC

# Detect linux

AMONLINUX:= $(shell uname -s | grep -o Linux)
ifeq "$(AMONLINUX)" "Linux"
LINUX=1
endif

# Detect cygwin

AMONCYGWIN:= $(shell uname -a | grep -o CYGWIN)
ifeq "$(AMONCYGWIN)" "CYGWIN"
WIN_MINGW32=1
endif

# Detect macs

AMONAMAC:= $(shell uname -s | grep -o Darwin)
ifeq "$(AMONAMAC)" "Darwin"
ONAMAC=1
endif

endif
endif
endif # Autodetect guard for operating system

# Finally, if none of those triggered,
ifndef LINUX
ifndef WIN_MINGW32
ifndef ONAMAC
$(error Unrecognized operating system in scripts/os_detect.inc)
endif
endif
endif

# Might need to guess what bitwidth target to build.
ifndef TARGET64
ifndef TARGET32
THEMACHINE:=$(shell uname -m)
ifeq ($(THEMACHINE),x86_64)
  TARGET64=1
else
  TARGET32=1
endif
endif
endif

