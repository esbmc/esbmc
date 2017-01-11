#!/bin/bash -e

set -x

if test ! -e esbmc; then
  echo "Please run from src/ dir";
  exit 1
fi

# Procedure:
# 1) Tell the user what's about to happen
# 2) Produce cleaned tarball via export-pub
# 3) For each target:
# 3a) Extract to an arbitary dir
# 3b) Run configure apropriately
# 3c) make
# 3d) Manufacture release tree
# 3e) Tar up
# 4) Clear up

# 1

echo "Will now produce a source distribution and compile into a release. Any local uncommitted changes will be ignored".

esbmcversion=`cat esbmc/version`
echo "ESBMC version: $esbmcversion"

if test `uname -m` != "x86_64"; then
  echo "You must build ESBMC releases on a 64 bit machine, sorry"
  exit 1
fi

if test `uname -s` != "Linux"; then
  echo "Not building on linux is not a supported operation; good luck"
  exit 1
fi

# 2

tmpfile=`mktemp /tmp/esbmc_release_XXXXXX`
srcdir=`mktemp -d /tmp/esbmc_release_XXXXXX`
builddir=`mktemp -d /tmp/esbmc_release_XXXXXX`
destdir=`mktemp -d /tmp/esbmc_release_XXXXXX`
here=`pwd`
releasedir="$here/.release"
mkdir $releasedir 2>/dev/null || true # Allow failure

fin () {
  rm $tmpfile
  rm -rf $srcdir
  rm -rf $builddir
  rm -rf $destdir
}

trap fin EXIT

scripts/export-pub.sh -n buildrelease $tmpfile

# 3

# 3a) Extract to an arbitary dir
tar -xzf $tmpfile -C $srcdir

do_build () {
  releasename=$1
  configureflags=$2
  buildhere=`pwd`

  # 3b) Run configure apropriately
  mkdir $builddir/$releasename
  cd $builddir/$releasename
  $srcdir/buildrelease/configure --prefix=/ $configureflags

  # Pause to get user to confirm solver sanity
  echo "Your solver configuration is shown above. Please confirm that the specified solver paths contain the correct solver versions for this release, and hit enter. Hit ctrl+c otherwise."
  read bees;

  # 3c) make
  # How many jobs?
  procs=`cat /proc/cpuinfo | grep ^processor | awk '{print $3;}' | sort | tail -n 1`
  procs=$(($procs + 1))
  make -j $procs
  # Will croak if that failed.

  # 3d) Manufacture release tree
  destname="$destdir/$releasename"
  mkdir $destname
  DESTDIR="$destname" make install


  # 3e) Tar up
  cd $destdir
  tar -czf "$releasedir/${releasename}.tgz" $releasename

  # Fini
  cd $buildhere
}

solver_opts="--disable-yices --disable-cvc4 --disable-mathsat --enable-z3 --enable-boolector"
x86flags="CXX=g++ CXXFLAGS=-m32 CFLAGS=-m32 LDFLFAGS=-m32"
flags="CXX=g++ CXXFLAGS=-DNDEBUG CFLAGS=-DNDEBUG --with-clang-libdir=$1 --with-llvm=$2"

do_build "esbmc-v${esbmcversion}-linux-64" "$flags $solver_opts"
if test $? != 0; then exit 1; fi

do_build "esbmc-v${esbmcversion}-linux-32" "$flags $solver_opts $x86flags"
if test $? != 0; then exit 1; fi

do_build "esbmc-v${esbmcversion}-linux-static-64" "$flags $solver_opts --enable-static-link"
if test $? != 0; then exit 1; fi

do_build "esbmc-v${esbmcversion}-linux-static-32" "$flags $solver_opts $x86flags --enable-static-link"
if test $? != 0; then exit 1; fi
