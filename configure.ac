AC_INIT([ESBMC], [1.22], [esbmc@ecs.soton.ac.uk])
AC_USE_SYSTEM_EXTENSIONS
AC_CONFIG_HEADER(ac_config.h)
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
LT_INIT
AC_CONFIG_MACRO_DIR([build-aux/m4])

AC_ARG_VAR([Z3PATH], [Path to Z3 installation directory])

AC_PROG_CC
AC_PROG_CXX
AC_PROG_AWK
AC_PROG_CPP
AC_PROG_SED
AC_PROG_YACC
AC_PROG_LEX
AC_PROG_MAKE_SET
AC_PROG_RANLIB

AC_LANG(C++)

AC_CHECK_LIB([m], [fabs])

AC_CHECK_HEADERS([boost/mpl/if.hpp])
AC_CHECK_HEADERS([boost/shared_ptr.hpp])
AC_CHECK_HEADERS([boost/crc.hpp])
AC_CHECK_HEADERS([boost/utility/enable_if.hpp])
AC_CHECK_HEADERS([boost/fusion/include/equal_to.hpp])

AC_HEADER_STDBOOL
AC_TYPE_UID_T
AC_C_INLINE
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T
AC_TYPE_MODE_T
AC_FUNC_OBSTACK
AC_TYPE_OFF_T
AC_TYPE_PID_T
AC_C_RESTRICT
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_CHECK_MEMBERS([struct stat.st_blksize])
AC_STRUCT_ST_BLOCKS
AC_CHECK_MEMBERS([struct stat.st_rdev])
AC_CHECK_DECLS([sys_siglist])
AC_STRUCT_TIMEZONE
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T
AC_CHECK_TYPES([ptrdiff_t])

AC_FUNC_CHOWN
AC_FUNC_ERROR_AT_LINE
AC_FUNC_FORK
AC_FUNC_FSEEKO
AC_FUNC_GETGROUPS
AC_FUNC_LSTAT_FOLLOWS_SLASHED_SYMLINK
AC_FUNC_MALLOC
AC_FUNC_MKTIME
AC_FUNC_MMAP
AC_FUNC_REALLOC
AC_FUNC_STRCOLL
AC_FUNC_STRERROR_R
AC_FUNC_STRNLEN
AC_FUNC_STRTOD
AC_CHECK_FUNCS([alarm atexit bzero clock_gettime dup2 euidaccess fchdir fdatasync fesetround floor ftruncate getcwd getdelim gethostname getpagesize getpass gettimeofday getusershell isascii lchown localeconv localtime_r mblen memchr memmove mempcpy memset mkdir mkfifo modf munmap pathconf pow putenv realpath rint rmdir rpmatch select setenv sethostname setlocale sqrt stime stpcpy strcasecmp strchr strcspn strdup strerror strncasecmp strndup strpbrk strrchr strspn strstr strtol strtoul strtoull strverscmp tzset])

LT_LIB_DLLOAD

AC_SUBST([ESBMC_CXXFLAGS],[["-Wall -Wextra -W -std=c++0x"]])

if test -z "$Z3PATH"; then
  echo "You must set Z3PATH. Or else.";
  exit 1;
fi

AC_HEADER_STDC

AC_CONFIG_FILES([Makefile big-int/Makefile util/Makefile langapi/Makefile solvers/Makefile solvers/prop/Makefile solvers/z3/Makefile goto-symex/Makefile goto-programs/Makefile pointer-analysis/Makefile ansi-c/Makefile ansi-c/cpp/Makefile ansi-c/headers/Makefile cpp/Makefile esbmc/Makefile])
AC_OUTPUT
