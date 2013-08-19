AC_INIT([ESBMC], [1.22], [esbmc@ecs.soton.ac.uk])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])

AC_PROG_CC
AC_PROG_CXX
AC_PROG_SED
AC_PROG_YACC
AC_PROG_LEX

AC_LANG(C++)

AC_CHECK_HEADERS([boost/mpl/if.hpp])
AC_CHECK_HEADERS([boost/shared_ptr.hpp])
AC_CHECK_HEADERS([boost/crc.hpp])
AC_CHECK_HEADERS([boost/utility/enable_if.hpp])
AC_CHECK_HEADERS([boost/fusion/include/equal_to.hpp])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
