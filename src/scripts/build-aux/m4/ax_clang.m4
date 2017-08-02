#
# AX_CLANG([ACTION-IF-FOUND,[ACTION-IF-NOT-FOUND]])
#
# If both clang and clang++ are detected
# - Sets shell variable HAVE_CLANG='yes'
# - If ACTION-IF-FOUND is defined: Runs ACTION-IF-FOUND
# - If ACTION-IF-FOUND is undefined:
#   - Runs AC_SUBST for variables CLANG and CLANGXX, setting them to the
#     corresponding paths.
#
# If not both clang and clang++ are detected
# - Sets shell variable HAVE_CLANG='no'
# - Runs ACTION-IF-NOT-FOUND if defined
#

AC_DEFUN([AX_CLANG],
[
    AC_ARG_WITH([clang],
        AS_HELP_STRING([--with-clang=PATH],
            [Force given directory for clang. Note that this will override library path detection, so use this parameter only if default library detection fails and you know exactly where your clang libraries are located.]),
            [
                if test -d "$withval"
                then
                    ac_clang_path="$withval"
                    AC_PATH_PROGS([CLANG],[clang],[],[$ac_clang_path/bin])
                else
                    AC_MSG_ERROR(--with-clang expected directory name)
                fi
            ],
            dnl defaults to $PATH
            [
                AC_PATH_PROGS([CLANG],[clang],[],[$PATH])
            ]
    )

    if test "x$CLANG" = "x"; then
        ifelse([$3], , :, [$3])
    fi

    dnl check clang version
    AC_MSG_CHECKING(if clang > $1)

    clangversion=0
    _version=$1
    clangversion=`$CLANG --version 2>/dev/null`
    if test "x$?" = "x0"; then
        clangversion=`echo "$clangversion" | tr '\n' ' ' | sed 's/^[[^0-9]]*\([[0-9]][[0-9.]]*[[0-9]]\).*$/\1/g'`
        clangversion=`echo "$clangversion" | sed 's/\([[0-9]]*\.[[0-9]]*\)\.[[0-9]]*/\1/g'`
        V_CHECK=`expr $clangversion \>= $_version`
        if test "$V_CHECK" != "1" ; then
            AC_MSG_WARN([Failure: ESBMC requires clang >= $1 but only found clang $clangversion.])
            ax_clang_ok='no'
        else
            ax_clang_ok='yes'
        fi
    else
        ax_clang_ok='no'
    fi

    if test "x$ax_clang_ok" = "xyes"; then
        AC_MSG_RESULT(yes ($clangversion))
    else
        AC_MSG_RESULT(no)
        ifelse([$3], , :, [$3])
    fi

    clang_base=/usr
    if test "$ac_clang_path" != ""; then
        clang_base=$ac_clang_path
    fi

    dnl Search for the includes
    AC_MSG_CHECKING(clang include directory)
    succeeded=no

    includesubdirs="$clang_base $clang_base/lib64/llvm-$clangversion $clang_base/lib/llvm-$clangversion"
    for includesubdir in $includesubdirs ; do
        if ls "$includesubdir/include/clang/Tooling/Tooling.h" >/dev/null 2>&1 ; then
            succeeded=yes
            clang_includes_path=$includesubdir/include
            break;
        fi
    done

    if test "$succeeded" != "yes" ; then
        AC_MSG_RESULT(no)
        ifelse([$3], , :, [$3])
    else
        AC_MSG_RESULT($clang_includes_path)
    fi

    dnl Now search for the libraries
    AC_MSG_CHECKING(clang library directory)
    succeeded=no

    dnl On 64-bit systems check for system libraries in both lib64 and lib.
    dnl The former is specified by FHS, but e.g. Debian does not adhere to
    dnl this (as it rises problems for generic multi-arch support).
    dnl The last entry in the list is chosen by default when no libraries
    dnl are found, e.g. when only header-only libraries are installed!
    libsubdirs="lib"
    ax_arch=`uname -m`
    case $ax_arch in
      x86_64)
        libsubdirs="lib64 libx32 lib lib64 lib64/llvm-$clangversion/lib libx32/llvm-$clangversion/lib lib/llvm-$clangversion/lib lib64/llvm-$clangversion/lib"
        ;;
      ppc64|s390x|sparc64|aarch64|ppc64le)
        libsubdirs="lib64 lib lib64 ppc64le lib64/llvm-$clangversion/lib lib/llvm-$clangversion/lib lib64/llvm-$clangversion/lib ppc64le/llvm-$clangversion/lib"
        ;;
    esac

    lib_ext="dylib"
    if test `uname` != "Darwin" ; then
        lib_ext="so"
    fi

    dnl Check the system location for clang libraries
    for libsubdir in $libsubdirs ; do
        if ls "$clang_base/$libsubdir/libclang"* >/dev/null 2>&1 ; then
            succeeded=yes
            clang_libs_path=$clang_base/$libsubdir
            break;
        fi
    done

    if test "$succeeded" != "yes" ; then
        AC_MSG_RESULT(no)
        ifelse([$3], , :, [$3])
    else
        AC_MSG_RESULT($clang_libs_path)
    fi

    dnl Look for clang libs
    clanglibs="Tooling Frontend Parse Sema Edit Analysis AST Lex Basic Driver Serialization"
    for lib in $clanglibs ; do
        AC_MSG_CHECKING(if we can find libclang$lib.$lib_ext)
        if ls "$clang_libs_path/libclang$lib"* >/dev/null 2>&1 ; then
            clang_LIBS="$clang_LIBS -lclang$lib"
            AC_MSG_RESULT(yes)
        else
            AC_MSG_NOTICE([Can't find libclang$lib])
            ifelse([$3], , :, [$3])
        fi
    done

    dnl Search if clang was shipped with a symbolic link call libgomp.so
    dnl We actually link with libgomp.so and this link breaks the old frontend
    if test -d "$withval" ; then
        AC_MSG_CHECKING(if $clang_libs_path/libgomp.so is present)
        if ls -L "$clang_libs_path/libgomp.so" >/dev/null 2>&1 ; then
            AC_MSG_ERROR([Found libgomp.so on $clang_libs_path. ESBMC is linked against the GNU libgomp and the one shipped with clang is known to cause issues on our tool. Please, remove it before continuing.])
        fi
        AC_MSG_RESULT(no)
    fi

    clang_CPPFLAGS="-I$clang_includes_path"
    clang_LDFLAGS="-L$clang_libs_path"

    CPPFLAGS_SAVED="$CPPFLAGS"
    CPPFLAGS="$CPPFLAGS $clang_CPPFLAGS"
    export CPPFLAGS

    LDFLAGS_SAVED="$LDFLAGS"
    LDFLAGS="$LDFLAGS $clang_LDFLAGS"
    export LDFLAGS

    LIBS_SAVED="$LIBS"
    LIBS="$LIBS $clang_LIBS"
    export LIBS

    if test "$succeeded" != "yes" ; then
        CPPFLAGS="$CPPFLAGS_SAVED"
        LDFLAGS="$LDFLAGS_SAVED"
        LIBS="$LIBS_SAVED"

        dnl execute ACTION-IF-NOT-FOUND (if present):
        ifelse([$3], , :, [$3])
    else
        AC_SUBST(clang_CPPFLAGS)
        AC_SUBST(clang_LDFLAGS)
        AC_SUBST(clang_LIBS)
        AC_DEFINE(HAVE_clang,,[define if the clang library is available])
        dnl execute ACTION-IF-FOUND (if present):
        ifelse([$2], , :, [$2])
    fi
])
