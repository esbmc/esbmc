# TODO:
# Sometimes libtool cracks up if we re-configure and don't make clean.
# (This may be a libtool feature)

# To bump the ESBMC version: edit esbmc/version to be whatever version required,
# and commit it. The m4 below will suck the version out of that file. Note that
# things will break if that file has an end-of-line -- use ":set binary" and
# ":set noeol" in vim to enforce that.

m4_define([esbmcversion], m4_include([esbmc/version]))
AC_INIT([ESBMC], [esbmcversion], [esbmc-users@googlegroups.com])
AC_DEFINE([ESBMC_VERSION], ["AC_PACKAGE_VERSION"], [ESBMC version number])
AC_SUBST([ESBMC_VERSION_STR], esbmcversion)
dnl Numerous things croak unless we do this immediately:
AC_CONFIG_AUX_DIR([scripts/build-aux])
AC_CONFIG_MACRO_DIRS([scripts/build-aux/m4])

AC_USE_SYSTEM_EXTENSIONS
AC_CONFIG_HEADER(ac_config.h:scripts/ac_config.in)
AM_INIT_AUTOMAKE([1.13 -Wall -Werror foreign])

AC_PROG_CC(clang gcc)
AC_PROG_CXX(clang++ g++)
AC_PROG_AWK
AC_PROG_CPP
AC_PROG_SED
AC_PROG_YACC
AC_PROG_LEX
AC_PROG_MAKE_SET

AC_CHECK_PROG(esbmc_has_yacc, $YACC, "yes", "no")
AS_IF([test "x$esbmc_has_yacc" = "xno"], AC_MSG_ERROR([Cannot find yacc]))
AC_CHECK_PROG(esbmc_has_lex, $LEX, "yes", "no")
AS_IF([test "x$esbmc_has_lex" = "xno"], AC_MSG_ERROR([Cannot find lex]))
AC_CHECK_PROG(esbmc_has_awk, $AWK, "yes", "no")
AS_IF([test "x$esbmc_has_awk" = "xno"], AC_MSG_ERROR([Cannot find awk]))
dnl Don't seem to be able to test for sed

dnl Transform version numbers into some integers...
AS_VAR_SET(ver_numcomponents, `echo esbmcversion | awk 'BEGIN{RS="."}{print $0;}' | wc -l | xargs `)
AS_IF([test "${ver_numcomponents}" != "4"], [
  AC_MSG_ERROR([Couldnt interpret esbmc version number in esbmc/version])
])
dnl Could use cut; too bad.
AS_VAR_SET(ver_majversion, `echo esbmcversion | awk 'BEGIN{FS="."}{print $1;}'`)
AS_VAR_SET(ver_minversion, `echo esbmcversion | awk 'BEGIN{FS="."}{print $2;}'`)
AS_VAR_SET(ver_pversion, `echo esbmcversion | awk 'BEGIN{FS="."}{print $3;}'`)
AS_ECHO("ESBMC version is ${ver_majversion}.${ver_minversion}.${ver_pversion}")

AS_IF([test "${ver_minversion}" -gt "255"], [
  AC_MSG_ERROR([Min version component is too large])
])
AS_IF([test "${ver_pversion}" -gt "255"], [
  AC_MSG_ERROR([Patch version component is too large])
])

AS_VAR_SET(ver_majhex, `printf "%X" ${ver_majversion}`)
AS_VAR_SET(ver_minhex, `printf "%02X" ${ver_minversion}`)
AS_VAR_SET(ver_phex, `printf "%02X" ${ver_pversion}`)
AS_VAR_SET(ver_hexstr, `printf "0x${ver_majhex}${ver_minhex}${ver_phex}"`)

AC_SUBST([ESBMC_VERSION_HEX], "${ver_hexstr}")

dnl Continue looking at our compilation situation

AX_CXX_COMPILE_STDCXX_11(noext,mandatory)

AS_VAR_SET(C2GOTO_LIBS, "")

AM_PROG_AR
AM_PROG_LEX

LT_INIT(static shared)
LTDL_INIT

AC_LANG(C++)

AC_CHECK_LIB([m], [fabs])

AC_CHECK_LIB([pthread], [pthread_create],
    [AS_VAR_APPEND(LIBS, " -lpthread ")]
)

AC_CHECK_LIB([gomp], [omp_get_num_threads],
    [AS_VAR_SET(LIBS, " -lgomp $LIBS")],
    [AC_MSG_WARN([No libgomp found: Z3 may want this])], []
)

AC_CHECK_LIB([dl], [dlerror],
    [AS_VAR_APPEND(LIBS, " -ldl")],
    [AC_MSG_WARN([No dl found: libgomp may want this])], []
)

AC_CHECK_LIB([gmp], [__gmpz_cmp],
    [AS_VAR_APPEND(LIBS, " -lgmp")],
    [AC_MSG_WARN([No libgmp found: various solvers require it])], []
)

# Define what to do when clang is enabled
AX_CLANG(3.8, [], AC_MSG_ERROR([Cannot find clang; please pass it in --with-clang argument]))
AX_LLVM(,AC_MSG_ERROR([Cannot find llvm; please pass it in --with-llvm argument]))
AC_SUBST([ESBMC_CLANG_SUBDIRS], " clang-c-frontend clang-cpp-frontend ")
AC_SUBST([ESBMC_CLANG_ARCHIVES], " ../clang-c-frontend/libclangcfrontend.la ../clang-cpp-frontend/libclangcppfrontend.la ../clang-c-frontend/headers/libclangheaders.la ../clang-c-frontend/AST/libclangcfrontendast.la")

AX_BOOST_BASE(1.47, [], AC_MSG_ERROR([Cannot find boost base dir; please pass it in --with-boost argument or set BOOST_ROOT]))
AX_BOOST_FILESYSTEM
AX_BOOST_SYSTEM

AS_VAR_APPEND(CPPFLAGS, " $BOOST_CPPFLAGS")
AS_VAR_APPEND(LDFLAGS, " $BOOST_LDFLAGS $BOOST_FILESYSTEM_LIB $BOOST_SYSTEM_LIB")

AC_CHECK_HEADERS([boost/mpl/if.hpp], [], AC_MSG_ERROR([No boost mpl found]))
AC_CHECK_HEADERS([boost/shared_ptr.hpp], [], AC_MSG_ERROR([No boost shared ptr found]))
AC_CHECK_HEADERS([boost/crc.hpp], [], AC_MSG_ERROR([No boost CRC library found]))
AC_CHECK_HEADERS([boost/utility/enable_if.hpp], [], AC_MSG_ERROR([No boost enable_if utility found]))
AC_CHECK_HEADERS([boost/fusion/include/equal_to.hpp], [], AC_MSG_ERROR([No boost fusion library found]))
AC_CHECK_HEADERS([sys/sendfile.h], AC_DEFINE([HAVE_SENDFILE_ESBMC], [], [Has sendfile header]), [])

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

# These macros check for "GNU Libc compatible functions", aka things that work
# in the way that the autoconf writers want rather than the way the spec
# specifies. No thanks.
# AC_FUNC_MALLOC
# AC_FUNC_REALLOC
AC_FUNC_CHOWN
AC_FUNC_ERROR_AT_LINE
AC_FUNC_FORK
AC_FUNC_FSEEKO
AC_FUNC_GETGROUPS
AC_FUNC_LSTAT_FOLLOWS_SLASHED_SYMLINK
AC_FUNC_MKTIME
AC_FUNC_MMAP
AC_FUNC_STRCOLL
AC_FUNC_STRERROR_R
AC_FUNC_STRNLEN
AC_FUNC_STRTOD
AC_CHECK_FUNCS([alarm atexit bzero clock_gettime dup2 euidaccess fchdir fdatasync fesetround floor ftruncate getcwd getdelim gethostname getpagesize getpass gettimeofday getusershell isascii lchown localeconv localtime_r mblen memchr memmove mempcpy memset mkdir mkfifo modf munmap pathconf pow putenv realpath rint rmdir rpmatch select setenv sethostname setlocale sqrt stime stpcpy strcasecmp strchr strcspn strdup strerror strncasecmp strndup strpbrk strrchr strspn strstr strtol strtoul strtoull strverscmp tzset])

AC_CHECK_FUNC(strlcat, [AC_DEFINE([HAVE_STRLCAT], [true], [strlcat present])], [])
AC_CHECK_FUNC(strlcpy, [AC_DEFINE([HAVE_STRLCPY], [true], [strlcpy present])], [])

LT_LIB_DLLOAD

AC_ARG_ENABLE(static-link, [AS_HELP_STRING([--enable-static-link], Build a static version of esbmc)])
AS_VAR_SET(ESBMC_LDFLAGS, "")
AS_VAR_SET(LIBESBMC_LDFLAGS, "")
AS_IF([test "${enable_static_link}" = "yes"], [
  AS_VAR_SET(ESBMC_LDFLAGS, " -all-static ")
], [
    AS_VAR_SET(LIBESBMC_LDFLAGS, " -shared ")
])
AC_SUBST([LIBESBMC_LDFLAGS], $LIBESBMC_LDFLAGS)
AC_SUBST([ESBMC_LDFLAGS], $ESBMC_LDFLAGS)

AC_ARG_ENABLE(werror, [AS_HELP_STRING([--disable-werror], Disable werror build flag)])

dnl on by default...
AC_ARG_ENABLE(enable-esbmc, [AS_HELP_STRING([--enable-esbmc], Build main ESBMC binary)])
AS_IF([test "${enable_esbmc}" = "yes"], [
  AC_SUBST([ESBMC_BINARY], "esbmc")
], [])

AC_ARG_ENABLE(enable-python, [AS_HELP_STRING([--enable-python], Build python support into ESBMC)])
AS_IF([test "${enable_python}" = "yes"], [
  dnl check for dependencies when building the static library
  AS_IF([test "${enable_static_link}" = "yes"], [
    AC_CHECK_LIB([expat], [XML_ExpatVersion],
      [AS_VAR_APPEND(LIBS, " -lexpat")],
      AC_MSG_ERROR([No libexpat found: libpython may want this]), []
    )

    AC_CHECK_LIB([util], [openpty],
      [AS_VAR_APPEND(LIBS, " -lutil")],
      AC_MSG_ERROR([No libutil found: libpython may want this]), []
    )

    AC_CHECK_LIB([z], [zlibVersion],
      [AS_VAR_APPEND(LIBS, " -lz")],
      AC_MSG_ERROR([No libz found: libpython may want this]), []
    )
  ])

  dnl Check for python. Hardcoded to python 3: I was going to make this
  dnl selectable, but then realised that everyone (even I) need a little
  dnl knudging towards python3 now and then.
  PKG_CHECK_MODULES([LIBPYTHON3], [python3], [
    dnl Add flags to build python in; enable build of libesbmc
    AS_VAR_APPEND(CXXFLAGS, " $LIBPYTHON3_CFLAGS -DWITH_PYTHON ")
    AS_VAR_APPEND(CFLAGS, " $LIBPYTHON3_CFLAGS -DWITH_PYTHON ")

    dnl Misery: debian wants -py34 on the end, autoconf library stuff doesn't
    dnl detect that. Attempt to build the version number by frobbing pkgconfig

    _PKG_CONFIG(PYTHON3_VERSION, modversion, python3)

    AS_VAR_SET(py3suffix, `echo $pkg_cv_PYTHON3_VERSION | sed -e 's/\.//g'`)
    AS_VAR_SET(py3libname, "boost_python-py${py3suffix}")

    dnl Alternately, if the user overrides with a --with-boost-python= spec,
    dnl the set the libname to that
    AC_ARG_WITH([boost-python-libname], [AS_HELP_STRING([--with-boost-python-libname], [Specify basename of boost-pythons shared object])], [
      AS_VAR_SET(py3libname, "${with_boost_python_libname}")],
    [])

    AS_VAR_SET(LIBS, " $LIBPYTHON3_LIBS $LIBS")
    AS_VAR_SET(C2GOTO_LIBS, " $LIBPYTHON3_LIBS $C2GOTO_LIBS")

    dnl Check if we can actually compile in boost-python. Needs to be after
    dnl python libs added to LIBS
    AC_CHECK_LIB(["${py3libname}"], [no_init],
    [
      dnl Alright, add the lib name to that
      AS_VAR_SET(LIBS, " -l${py3libname} $LIBS")
      AS_VAR_SET(C2GOTO_LIBS, " -l${py3libname} $C2GOTO_LIBS")
    ],
    [
      dnl Fedora just has -lboost_python3
      AC_CHECK_LIB(["boost_python3"], [no_init],
      [
        AS_VAR_SET(LIBS, " -lboost_python3 $LIBS")
        AS_VAR_SET(C2GOTO_LIBS, " -lboost_python3 $C2GOTO_LIBS")
      ],
      
        AC_MSG_ERROR([Python supported requested but boost-python not found. Alternately: packaging sketchyness means I couldn't deduce the base soname and you can specify it with --with-boost-python-libname=boost_python-foobar'])
      )
    ])

    AC_SUBST([ESBMC_PYTHON_OBJ], "python.lo")
  ],
    AC_MSG_ERROR(Asked for python support but pkgconfig finds no python library. You need to install python 3 development packages.)
  )
], [
    AC_SUBST([ESBMC_PYTHON_OBJ], "")
])

dnl If --enable-libesbmc was supplied, then enable building of libesbmc.
AS_IF([test "${enable_libesbmc}" = "yes"], [
  AC_SUBST([ESBMC_LIB], "libesbmc.la")
], [
  AC_SUBST([ESBMC_LIB], "")
])

AS_VAR_SET(BASE_CXXFLAGS, "-Wall -Wextra -W -std=c++11 $ESBMC_CLANG_CXXFLAGS ")
dnl A necessary evil due to the presence of rubbish in boost::python's headers.
dnl local typedefs because that's how boost python works
dnl deprecated decls because it uses auto ptrs
dnl field initializers due to a, uh, totally legitimate initializer mismatch
dnl       with later versions of python headers
dnl String aliasing... the following knackers that:
dnl       return (*(void**)m_result.bytes == Py_None)
AS_VAR_APPEND(BASE_CXXFLAGS, " -Wno-unused-local-typedefs -Wno-deprecated-declarations -Wno-missing-field-initializers -Wno-strict-aliasing $ESBMC_CLANG_CFLAGS ")

dnl # Do not turn -Werror off: your code is wrong, not -Werror.
dnl Unless of course you're on a mac, in which case clang will frequently
dnl generate new warnings, and you probably want it off. This should be used
dnl with care in that case.
AS_IF([test "x$enable_werror" = "xno"], [
  AC_SUBST([ESBMC_CXXFLAGS],[["$BASE_CXXFLAGS"]])
  AC_SUBST([ESBMC_CFLAGS],[["-Wall -Wextra -W"]])
], [
  dnl not disabled -> enable werror
  AC_SUBST([ESBMC_CXXFLAGS],[["$BASE_CXXFLAGS -Werror"]])
  AC_SUBST([ESBMC_CFLAGS],[["-Wall -Wextra -W -Werror"]])
])

AC_SUBST([ESBMC_CXXFLAGS_WARNINGS],[["-Wall -Wextra -W -std=c++11"]])

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
  dnl $8 = Macro for additional local library config
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
    # Prepend library to LIBS -- this is because the link ordering is important,
    # and the library might depend on previously detected libraries already in
    # LIBS
    LIBS="[-L]path/libdir -l$4 $ESBMC_SOLVER_LIBS $LIBS"
    # Indirectly call local-config macro
    indir($8, path, "path/libdir", LIBS)
    AC_CHECK_LIB($4, $5, [
      AS_VAR_SET(esbmc_check_solver_libthere_${esbmc_path_name}, yes)
      AS_VAR_SET(esbmc_check_solver_libdir_${esbmc_path_name}, libdir)
      AS_VAR_SET(esbmc_found_solver_$3_path, path)
      AS_VAR_SET(esbmc_found_solver_$3, yes)
      AS_VAR_APPEND(esbmc_list_of_solvers, " $3")
      AS_VAR_APPEND(esbmc_list_of_backends, [" "$3/'lib'$3'.la'])
      AS_VAR_APPEND(ESBMC_SOLVER_LIBS, " [-L]path/libdir -l$4 ")
      indir($8, path, "path/libdir", ESBMC_SOLVER_LIBS)
      AC_SUBST([ESBMC_ENABLE_$3], 1)
      AS_ECHO("Found solver at path/libdir")
    ], [
      AC_SUBST([ESBMC_ENABLE_$3], 0)
    ], [])
    dnl Reset LIBS. Intermediate programs should not depend on solvers
    AS_VAR_SET(LIBS, $esbmc_check_solver_old_libs)

    # Don't cache the library check, seeing how we're changing the env.
    AS_UNSET(ac_cv_lib_$4_$5)


    ])
    ])
    ])

     # Unset flags if no solver found yet
    AS_IF([test "$[esbmc_found_solver_]$3" != "yes"], [
      CXXFLAGS=$esbmc_check_solver_old_cxxflags
      CPPFLAGS=$esbmc_check_solver_old_cppflags
      ])
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
  dnl $8 = Macro for additional local library config
  dnl XXX output
  AC_ARG_WITH([$3], [AS_HELP_STRING([--with-$3], [Enable $3 solver])], [],
              [with_$3=yes])

  AS_IF([test "x$with_$3" != xno], [

    AS_IF([test "x$with_$3" != xyes], [
      AS_VAR_SET(esbmc_with_dir_path, "$with_$3")
      ESBMC_CHECK_SOLVER([$esbmc_with_dir_path], [$2], $3, $4, $5, $6, [$7], $8)
      AS_IF([test "$esbmc_found_solver_$3" != "yes"],
            AC_MSG_ERROR(["Path to $3 given on command line (with --with-$3); but can't use $3 installation there"]))
      ], [
        ESBMC_CHECK_SOLVER([$1], [$2], $3, $4, $5, $6, [$7], $8)
      ]
      )
    ])
  ])

m4_define(esbmc_all_solvers, [z3, boolector, yices, cvc4, mathsat])
m4_foreach(thesolver, [esbmc_all_solvers], [
  AC_ARG_ENABLE(thesolver,
              [AS_HELP_STRING([--disable-thesolver], Disable thesolver solver)])
  AS_IF([test "x$[enable_]thesolver" = "xno"], [
           AS_VAR_SET([esbmc_disable_solver_]thesolver, "yes")
        ])
  ])

dnl Define all of these to be not enabled by default
m4_foreach(thesolver, [esbmc_all_solvers], [
  m4_define([str], [ESBMC_ENABLE_])
  m4_append([str], thesolver)
  AC_SUBST([str], 0)
  undefine([str])
])

m4_define(z3_build_headers, [include/z3.h, include/z3++.h])
m4_define(z3_possible_locs, [$SATDIR64/z3, $SATDIR32/z3, $SATDIR/z3, $HOME/z3, $Z3DIR])
m4_define(z3_possible_libdirs, [lib, bin, x64])

dnl Only boolector 1.6 and later are supported -- prior versions of boolector
dnl do not have a structure for knowing where lingeling / minisat are.

m4_define(btor_build_headers, [boolector/src/boolector.h])
m4_define(btor_possible_locs, [$SATDIR64/boolector, $SATDIR32/boolector, $SATDIR/boolector, $HOME/boolector])
m4_define(btor_possible_libdirs, [boolector/build])

m4_define(yices_build_headers, [include/yices.h])
m4_define(yices_possible_locs, [$SATDIR64/yices, $SATDIR32/yices, $SATDIR/yices, $HOME/yices])
m4_define(yices_possible_libdirs, [lib])

m4_define(cvc4_build_headers, [include/cvc4/cvc4.h])
m4_define(cvc4_possible_locs, [$SATDIR64/cvc4, $SATDIR32/cvc4, $SATDIR/cvc4, $HOME/cvc4])
m4_define(cvc4_possible_libdirs, [lib])

m4_define(mathsat_build_headers, [include/mathsat.h])
m4_define(mathsat_possible_locs, [$SATDIR64/mathsat, $SATDIR32/mathsat, $SATDIR/mathsat, $HOME/mathsat])
m4_define(mathsat_possible_libdirs, [lib])

dnl Random dollars are shoehorned into these names to ensure that M4 does not
dnl rename them as they are passed down to the point where they are actually
dnl used -- to set additional command line options for a particular solver
dnl candiate location
m4_define([$$ESBMC_NOOP_CONFIG], [])

ESBMC_SOLVER_LIBS=""

dnl Lingeling gets it's own special position in the command line because it's
dnl an intermediate library that a solver depends on, and thus has to go after
dnl any solver -l$solver directives
m4_define([$$ESBMC_BOOLECTOR_EXTRA_CONFIG], [
  dnl $1 = Path we're looking at as being the boolector installation
  dnl $2 = libdir path
  dnl $3 = Variable we're pumping this config into
  AS_VAR_APPEND($3, " [-L]$1/lingeling ")
  AS_VAR_APPEND($3, " -llgl ")
])dnl

m4_define([$$ESBMC_YICES_EXTRA_CONFIG], [
  dnl $1 = Path we're looking at as being the yices installation
  dnl Yices ships with 'libyices.2.x.y', but no set of appropriate symlinks.
  dnl So linking against it is a pain. Detect cricumstances where the
  dnl appropriate version symlinks are not installed, and bail.
  AS_IF([test -d $1], [
    AS_IF([test ! -z "`ls $2 | grep libyices.so.2*`"], [
      AS_IF([test ! -e "$2/libyices.so"],
        AC_MSG_ERROR([Found yices but cannot link against it. A shared object exists in $2; however you need to symlink a "libyices.so" to it to work. Or, specify --disable-yices])
        exit 1
      )
    ])
  ])
])dnl

esbmc_list_of_solvers="smtlib";
esbmc_list_of_backends="./smtlib/libsmtlib.la ";

AS_IF([test "$esbmc_disable_solver_z3" != "yes"], [
ESBMC_OPTIONAL_SOLVER([z3_possible_locs], [z3_build_headers], z3, z3, Z3_mk_context, include, [z3_possible_libdirs], $$ESBMC_NOOP_CONFIG)
])

AS_IF([test "$esbmc_disable_solver_boolector" != "yes"], [
ESBMC_OPTIONAL_SOLVER([btor_possible_locs], [btor_build_headers], boolector, boolector, boolector_new, boolector/src, [btor_possible_libdirs], $$ESBMC_BOOLECTOR_EXTRA_CONFIG)
])

AS_IF([test "$esbmc_disable_solver_yices" != "yes"], [
ESBMC_OPTIONAL_SOLVER([yices_possible_locs], [yices_build_headers], yices, yices, yices_init, include, [yices_possible_libdirs], $$ESBMC_YICES_EXTRA_CONFIG)
])

AS_IF([test "$esbmc_disable_solver_cvc4" != "yes"], [
dnl NB: this checks for 'sleep' in libcvc4 (which resolves) because autoconf
dnl does not enjoy the idea of testing for C++ / mangled symbols
ESBMC_OPTIONAL_SOLVER([cvc4_possible_locs], [cvc4_build_headers], cvc4, cvc4, sleep, include, [cvc4_possible_libdirs], $$ESBMC_NOOP_CONFIG)
])

AS_IF([test "$esbmc_disable_solver_mathsat" != "yes"], [
ESBMC_OPTIONAL_SOLVER([mathsat_possible_locs], [mathsat_build_headers], mathsat, mathsat, msat_create_config, include, [mathsat_possible_libdirs], $$ESBMC_NOOP_CONFIG)
])

dnl Temporarily, while things remain unconfigured
AC_SUBST([ESBMC_ENABLE_minisat], 0)

# Define a list of solvers that we have
AC_DEFINE_UNQUOTED([ESBMC_AVAILABLE_SOLVERS], ["$esbmc_list_of_solvers"], [List of ESBMC headers available to us])
AC_SUBST([ESBMC_AVAILABLE_SOLVERS], $esbmc_list_of_solvers)
AC_SUBST([ESBMC_SOLVER_BACKENDS], ["$esbmc_list_of_backends"])
AC_SUBST([ESBMC_SOLVER_DIRS], ["$esbmc_list_of_solvers"])
AC_SUBST([ESBMC_SOLVER_LIBS], "$ESBMC_SOLVER_LIBS")

AC_CONFIG_FILES([Makefile big-int/Makefile util/Makefile langapi/Makefile solvers/Makefile solvers/prop/Makefile solvers/smt/Makefile solvers/z3/Makefile solvers/boolector/Makefile solvers/yices/Makefile solvers/cvc4/Makefile solvers/mathsat/Makefile solvers/smtlib/Makefile goto-symex/Makefile goto-programs/Makefile pointer-analysis/Makefile ansi-c/Makefile ansi-c/cpp/Makefile cpp/Makefile cpp/library/Makefile clang-c-frontend/Makefile clang-c-frontend/headers/Makefile clang-c-frontend/AST/Makefile clang-cpp-frontend/Makefile c2goto/headers/Makefile c2goto/Makefile esbmc/Makefile solvers/solver_config.h esbmc/version.h])
AC_OUTPUT

AS_ECHO(["----------------------------------------"])
AS_ECHO(["Your solver configuration is below"])
AS_ECHO(["To override, pass --with-solvername=path"])
AS_ECHO(["----------------------------------------"])
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
