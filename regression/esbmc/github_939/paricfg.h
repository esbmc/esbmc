/*  This file was created by Configure. Any change made to it will be lost
 *  next time Configure is run. */
#ifndef __PARICFG_H__
#define __PARICFG_H__
#define UNIX
#define GPHELP "\"/usr/local/bin/gphelp\""
#define GPDATADIR "/usr/local/share/pari"
#define SHELL_Q '\''

#define PARIVERSION "GP/PARI CALCULATOR Version 2.15.2 (released)"
#define PARIINFO "amd64 running linux (x86-64/GMP-%s kernel) 64-bit version"
#define PARI_VERSION_CODE 134914
#define PARI_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))
#define PARI_VERSION_SHIFT 8
#define PARI_VCSVERSION ""
#define PARI_MT_ENGINE "single"

#define PARI_DOUBLE_FORMAT -
#define GCC_VERSION "/usr/local/bin/afl-gcc-fast"
#define ASMINLINE

/*  Location of GNU gzip program (enables reading of .Z and .gz files). */
#define GNUZCAT
#define ZCAT "/usr/bin/gzip -dc"


#define GP_MIME_OPEN "xdg-open"

#define LONG_IS_64BIT
#define HAS_SSE2
#define HAS_EXP2
#define HAS_LOG2
#define HAS_RINT
#define HAS_ISATTY
#define HAS_ALARM
#define HAS_SYSTEM
#define USE_GETRUSAGE 1
#define USE_GETTIMEOFDAY 1
#define HAS_SIGACTION
#define HAS_WAITPID
#define HAS_GETENV
#define HAS_SETSID
#define HAS_DLOPEN
#define STACK_CHECK
#define HAS_VSNPRINTF
#define HAS_TIOCGWINSZ
#define HAS_STRFTIME
#define HAS_STAT
#define HAS_MMAP
#endif
