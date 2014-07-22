AC_INIT([ESBMC], [1.23.99], [esbmc@ecs.soton.ac.uk])
AC_DEFINE([ESBMC_VERSION], ["1.23.99"], [ESBMC version number])

AC_USE_SYSTEM_EXTENSIONS
AC_CONFIG_HEADER(ac_config.h)
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([1.13 -Wall -Werror foreign])
AC_CONFIG_MACRO_DIR([build-aux/m4])

AC_ARG_VAR([Z3PATH], [Path to Z3 installation directory])

AC_PROG_CC(clang gcc)
AC_PROG_CXX(clang++ g++)
AC_PROG_AWK
AC_PROG_CPP
AC_PROG_SED
AC_PROG_YACC
AC_PROG_LEX
AC_PROG_MAKE_SET
AC_PROG_RANLIB

AX_CXX_COMPILE_STDCXX_11(noext,mandatory)

AM_PROG_AR
AM_PROG_LEX

LT_INIT

AC_LANG(C++)

AC_CHECK_LIB([m], [fabs])

AC_CHECK_HEADERS([boost/mpl/if.hpp], [], [AC_MSG_ERROR([No boost mpl found])])
AC_CHECK_HEADERS([boost/shared_ptr.hpp], [], [AC_MSG_ERROR([No boost shared ptr found])])
AC_CHECK_HEADERS([boost/crc.hpp], [], [AC_MSG_ERROR([No boost CRC library found])])
AC_CHECK_HEADERS([boost/utility/enable_if.hpp], [], [AC_MSG_ERROR([No boost enable_if utility found])])
AC_CHECK_HEADERS([boost/fusion/include/equal_to.hpp], [], [AC_MSG_ERROR([No boost fusion library found])])
AC_CHECK_HEADERS([sys/sendfile.h], [AC_DEFINE([HAVE_SENDFILE], [], [Has sendfile header])], [])

AC_CHECK_HEADERS([unistd.h], [AC_DEFINE([HAVE_UNISTD], [], [Has unistd.h header])], [])

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

AC_CHECK_FUNC(strlcat, [AC_DEFINE([HAVE_STRLCAT], [true], [strlcat present])], [])
AC_CHECK_FUNC(strlcpy, [AC_DEFINE([HAVE_STRLCPY], [true], [strlcpy present])], [])

LT_LIB_DLLOAD

AC_SUBST([ESBMC_CXXFLAGS],[["-Wall -Wextra -W -Werror -std=c++0x"]])
AC_SUBST([ESBMC_CFLAGS],[["-Wall -Wextra -W"]])

AC_SUBST([ESBMC_CXXFLAGS_WARNINGS],[["-Wall -Wextra -W -std=c++0x"]])

AC_HEADER_STDC

AC_DEFUN([ESBMC_FINDSOLVER_ADDPREFIX], [m4_foreach(thevar, [$2], [$1/thevar$3])]dnl
  dnl $1 = prefix
  dnl $2 = list of filenames, comma separated
  dnl $3 = separator
  dnl evaluates to list of files, not comma separated
  dnl
)

AC_DEFUN([ESBMC_CHECK_FILES], [
  dnl $1 = Path prefix
  dnl $2 = File list
  dnl $3 = output var to set
  AC_CHECK_FILES(ESBMC_FINDSOLVER_ADDPREFIX($1, [$2], [ ]))
  $3=yes
  for fname in ESBMC_FINDSOLVER_ADDPREFIX($1, [$2], [ ]); do
    goodfname=AS_TR_SH($fname)
    AS_VAR_COPY([var], [ac_cv_file_$goodfname])
    AS_IF([test $var = no], [$3=no;])
  done
  ]
)

dnl Yes, I know
AC_DEFUN([ESBMC_CHECK_HEADERS], [
  dnl $1 = Path prefix
  dnl $2 = File list
  dnl $3 = output var to set
  AC_CHECK_HEADERS(ESBMC_FINDSOLVER_ADDPREFIX($1, [$2], [ ]))
  $3=yes
  for fname in ESBMC_FINDSOLVER_ADDPREFIX($1, [$2], [ ]); do
    goodfname=AS_TR_SH($fname)
    AS_VAR_COPY([var], [ac_cv_header_$goodfname])
    AS_IF([test $var = no], [$3=no;])
  done
  ]
)

AC_DEFUN([ESBMC_CHECK_SOLVER], [
  dnl $1 = Paths
  dnl $2 = Headers
  dnl $3 = solvername
  dnl $4 = The base solver library file, i.e. z3 for -lz3
  dnl $5 = A function in the library to test for presence of
  dnl $6 = include subpath
  dnl $7 = list of potential library subpaths
  dnl XXX output
  AS_VAR_SET(esbmc_found_solver_$3, no)
  m4_foreach(path, [$1], [
    dnl Skip if we already have one
    AS_VAR_COPY(alreadygotone, esbmc_found_solver_$3)
    AS_IF([test $alreadygotone = "no"], [

    esbmc_path_name=AS_TR_SH($3_$path)
    AS_VAR_SET(esbmc_check_solver_${esbmc_path_name}, no)
    ESBMC_CHECK_FILES(path, [$2], has_files)
    AS_IF([test $has_files = yes], [
    esbmc_check_solver_old_cppflags=$CPPFLAGS
    esbmc_check_solver_old_cxxflags=$CXXFLAGS
    AS_VAR_APPEND(CPPFLAGS, " -I path/$6 ")
    AS_VAR_APPEND(CXXFLAGS, " -I path/$6 ")
    ESBMC_CHECK_HEADERS(path, [$2], has_headers)
    AS_IF([test $has_headers = yes], [

    AS_VAR_SET(esbmc_check_solver_libthere_${esbmc_path_name}, no)
    AS_VAR_SET(esbmc_check_solver_libdir_${esbmc_path_name}, "")
    m4_foreach(libdir, [$7], [

    # Only test if we haven't found something
    AS_VAR_COPY(alreadydone, esbmc_check_solver_libthere_${esbmc_path_name})
    AS_IF([test $alreadydone = "no"], [

    esbmc_check_solver_old_libs=$LIBS
    AS_VAR_APPEND(LIBS, " -L path/libdir ")
    AC_CHECK_LIB($4, $5, [
      AS_VAR_SET(esbmc_check_solver_libthere_${esbmc_path_name}, yes)
      AS_VAR_SET(esbmc_check_solver_libdir_${esbmc_path_name}, libdir)
      AS_VAR_SET(esbmc_found_solver_$3_path, path)
      AS_VAR_SET(esbmc_found_solver_$3, yes)
      AS_ECHO("Found solver at path/libdir")
    ], [
      dnl Reset LIBS if not found
      AS_VAR_SET(LIBS, $esbmc_check_solver_old_libs)
    ], [])

    # Don't cache the library check, seeing how we're changing the env.
    AS_UNSET(ac_cv_lib_$4_$5)


    ])
    ])
    ])
    CXXFLAGS=$esbmc_check_solver_old_cxxflags
    CPPFLAGS=$esbmc_check_solver_old_cppflags
    ])
    ])
  ])
])

m4_define(ESBMC_OPTIONAL_SOLVER, [
  dnl $1 = Paths
  dnl $2 = Headers
  dnl $3 = solvername
  dnl $4 = The base solver library file, i.e. z3 for -lz3
  dnl $5 = A function in the library to test for presence of
  dnl $6 = include subpath
  dnl $7 = list of potential library subpaths
  dnl XXX output
  AC_ARG_WITH([$3], [AS_HELP_STRING([--with-$3], [Enable $3 solver])], [],
              [with_$3=yes])

  AS_IF([test "x$with_$3" != xno], [

    AS_IF([test "x$with_$3" != xyes], [
      AS_VAR_SET(esbmc_with_dir_path, "$with_$3")
      ESBMC_CHECK_SOLVER([$esbmc_with_dir_path], [$2], $3, $4, $5, $6, [$7])
      AS_IF([test "$esbmc_found_solver_$3" != "yes"],
            [ AC_MSG_ERROR(["Path to $3 given on command line (with --with-$3), but can't use $3 installation there"])])
      ], [
        ESBMC_CHECK_SOLVER([$1], [$2], $3, $4, $5, $6, [$7])
      ]
      )
    ])
  ])

m4_define(esbmc_all_solvers, [z3])

m4_define(z3_build_headers, [include/z3.h, include/z3++.h])
m4_define(z3_possible_locs, [$SATDIR64/z3, $SATDIR/z3, $HOME/z3, $Z3DIR])
m4_define(z3_possible_libdirs, [lib, bin, x64])


ESBMC_OPTIONAL_SOLVER([z3_possible_locs], [z3_build_headers], z3, z3, Z3_mk_context, include, [z3_possible_libdirs])

AC_CONFIG_FILES([Makefile big-int/Makefile util/Makefile langapi/Makefile solvers/Makefile solvers/prop/Makefile solvers/z3/Makefile goto-symex/Makefile goto-programs/Makefile pointer-analysis/Makefile ansi-c/Makefile ansi-c/cpp/Makefile ansi-c/headers/Makefile cpp/Makefile esbmc/Makefile])
AC_OUTPUT

AS_ECHO(["---------------------------------"])
AS_ECHO(["Your solver configuration is thus"])
AS_ECHO(["---------------------------------"])
esbmc_found_a_solver=no
m4_foreach(thesolver, [esbmc_all_solvers], [
  AS_IF([test "$[esbmc_found_solver_]thesolver" = "yes"], [
    esbmc_found_a_solver=yes
    AS_ECHO("AS_HELP_STRING([thesolver], $[esbmc_found_solver_]thesolver[_path])")
  ]
  )
])
AS_IF([test "$esbmc_found_a_solver" = "no"], [
   AS_ECHO("WARNING: No solvers found. It's quite likely that this isn't what you want.")
   AS_ECHO("")
   AS_ECHO(["Configure \$[]SATDIR to point at a directory containing sat solvers, set an appropriate environmental variable, or pass a path to a solver to configure with --with-solvername=path"])
   AS_ECHO("")
   AS_ECHO(["You can still use ESBMC using the SMTLIB printer, however you only be able to output formulae to a file, or connect the SMTLIB printer to a solver process."])
   ])
