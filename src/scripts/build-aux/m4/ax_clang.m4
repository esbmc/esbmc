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
        AC_ARG_WITH([clang-libdir],
                AS_HELP_STRING([--with-clang-libdir=LIB_DIR],
                [Force given directory for clang libraries. Note that this will override library path detection, so use this parameter only if default library detection fails and you know exactly where your clang libraries are located.]),
                [
                if test -d "$withval"
                then
                        ac_clang_lib_path="$withval"
                else
                        AC_MSG_ERROR(--with-clang-libdir expected directory name)
                fi
                ],
                [ac_clang_lib_path=""]
        )

    AC_MSG_CHECKING(for clang >= $1)
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
        libsubdirs="lib64 libx32 lib lib64"
        ;;
      ppc64|s390x|sparc64|aarch64|ppc64le)
        libsubdirs="lib64 lib lib64 ppc64le"
        ;;
    esac

    dnl allow for real multi-arch paths e.g. /usr/lib/x86_64-linux-gnu. Give
    dnl them priority over the other paths since, if libs are found there, they
    dnl are almost assuredly the ones desired.
    AC_REQUIRE([AC_CANONICAL_HOST])
    libsubdirs="lib/${host_cpu}-${host_os} $libsubdirs"

    case ${host_cpu} in
      i?86)
        libsubdirs="lib/i386-${host_os} $libsubdirs"
        ;;
    esac

    _version=0
    dnl first we check the system location for clang libraries and version
    if test "$ac_clang_lib_path" != ""; then
        clang_includes_path=$ac_clang_lib_path/include/clang
        for libsubdir in $libsubdirs ; do
            if ls "$ac_clang_lib_path/$libsubdir/libclang"* >/dev/null 2>&1 ; then break; fi
        done

        for i in `ls -d $ac_clang_lib_path/$libsubdir/libclang.so.* 2>/dev/null`; do
            _version_tmp=`echo $i | sed "s#$ac_clang_lib_path/$libsubdir/##" | sed 's/libclang.so.//'`
            V_CHECK=`expr $_version_tmp \> $_version`
            if test "$V_CHECK" != "1" ; then
                    continue
            fi

            _version=$_version_tmp
            succeeded=yes

            clang_libs_path=$ac_clang_lib_path/$libsubdir
            break;
        done
    elif test "$cross_compiling" != yes; then
        for ac_clang_lib_path_tmp in /usr /usr/local /opt /opt/local ; do
            if test -d "$ac_clang_lib_path_tmp/include/clang" && test -r "$ac_clang_lib_path_tmp/include/clang"; then
                for libsubdir in $libsubdirs ; do
                    if ls "$ac_clang_lib_path_tmp/$libsubdir/libclang"* >/dev/null 2>&1 ; then break; fi
                done

                for i in `ls -d $ac_clang_lib_path_tmp/$libsubdir/libclang.so.* 2>/dev/null`; do
                    _version_tmp=`echo $i | sed "s#$ac_clang_lib_path_tmp/$libsubdir/##" | sed 's/libclang.so.//'`
                    V_CHECK=`expr $_version_tmp \> $_version`
                    if test "$V_CHECK" != "1" ; then
                            continue
                    fi

                    _version=$_version_tmp
                    succeeded=yes

                    clang_libs_path=$ac_clang_lib_path_tmp/$libsubdir
                    clang_includes_path=$ac_clang_lib_path_tmp/include
                    break;
                done
            fi
        done
    fi

    if test "$succeeded" != "yes" ; then
        AC_MSG_RESULT(no)
        ifelse([$3], , :, [$3])
    else
        AC_MSG_RESULT(yes)
    fi

    dnl Look for clang libs
    clanglibs="Tooling Driver Frontend Parse Serialization Sema Analysis Edit Lex AST Basic"
    for lib in $clanglibs ; do
        AC_MSG_CHECKING(if we can find libclang$lib)
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
    AC_MSG_CHECKING(if $clang_libs_path/libgomp.so is present)
    if ls -L "$clang_libs_path/libgomp.so" >/dev/null 2>&1 ; then
        AC_MSG_ERROR([Found libgomp.so on $clang_libs_path. ESBMC is linked against the GNU libgomp and the one shipped with clang is known to cause issues on our tool. Please, remove it before continuing.])
    fi
    AC_MSG_RESULT(no)

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
        if test "$_version" = "0" ; then
            AC_MSG_NOTICE([[We could not detect the clang libraries (version $clang_lib_version_req or higher). If you have a staged clang library (still not installed) please specify \$CLANG_ROOT in your environment and do not give a PATH to --with-clang option.]])
        else
            CPPFLAGS="$CPPFLAGS_SAVED"
            LDFLAGS="$LDFLAGS_SAVED"
            LIBS="$LIBS_SAVED"

            # execute ACTION-IF-NOT-FOUND (if present):
            ifelse([$3], , :, [$3])
        fi
    else
        AC_SUBST(clang_CPPFLAGS)
        AC_SUBST(clang_LDFLAGS)
        AC_SUBST(clang_LIBS)
        AC_DEFINE(HAVE_clang,,[define if the clang library is available])
        # execute ACTION-IF-FOUND (if present):
        ifelse([$2], , :, [$2])
    fi
])
