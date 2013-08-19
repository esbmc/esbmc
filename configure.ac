AC_INIT([ESBMC], [1.22], [esbmc@ecs.soton.ac.uk])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])

AC_ARG_VAR([Z3PATH], [Path to Z3 installation directory])

AC_PROG_CC
AC_PROG_CXX
AC_PROG_SED
AC_PROG_YACC
AC_PROG_LEX
AC_PROG_RANLIB

AC_LANG(C++)

AC_CHECK_HEADERS([boost/mpl/if.hpp])
AC_CHECK_HEADERS([boost/shared_ptr.hpp])
AC_CHECK_HEADERS([boost/crc.hpp])
AC_CHECK_HEADERS([boost/utility/enable_if.hpp])
AC_CHECK_HEADERS([boost/fusion/include/equal_to.hpp])

AC_SUBST([AM_CXXFLAGS],[["-Wall -Wextra -W -std=c++0x -I `pwd` -I `pwd`/util"]])

if test -z "$Z3PATH"; then
  echo "You must set Z3PATH. Or else.";
  exit 1;
fi

AC_CONFIG_FILES([Makefile big-int/Makefile util/Makefile langapi/Makefile solvers/Makefile solvers/prop/Makefile solvers/z3/Makefile goto-symex/Makefile])
AC_OUTPUT
