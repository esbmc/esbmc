#!/bin/bash

function checksanity() {
  # You need a 64 bit machine to fully build a release
  if test `uname -m` != "x86_64"; then
    echo "Please run release.sh on a 64 bit machine"
    exit 1
  fi

  # You also need to be running it in the root ESBMC dir
  stat .git > /dev/null 2>/dev/null
  if test $? != 0; then
    echo "Please run release.sh in the root dir of ESBMC"
    exit 1
  fi

  # Check to see whether or not there's an instance for
  # this version in release notes
  # (Start by removing leading v)
  vernum=`echo $1 | sed s#v\(.*\)#\1#`
  grep "\*\*\*.*$vernum.*\*\*\*" ./scripts/release-notes.txt > /dev/null 2>&1
  if test $? != 0; then
    echo "Can't find an entry for $1 in release-notes.txt; you need to write one"
    exit 1
  fi
}

checksanity

while getopts ":3:6:2:5:r:" opt; do
  case $opt in
    3)
      satdir32=$OPTARG
      ;;
    2)
      satdir32compat=$OPTARG
      ;;
    6)
      satdir64=$OPTARG
      ;;
    5)
      satdir64compat=$OPTARG
      ;;
    r)
      targetrefname=$OPTARG
      ;;
    \?)
      echo "Invalid option -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires argument" >&2
      exit 1
      ;;
  esac
done

function setdefaultsatdir () {
  curname=$1
  defaultname=$2
  valuename=$3
  optname=$4

  if test -z $curname; then
    if test -z $defaultname; then
      echo "Can't autodetect a value for $valuename, give it with option $optname" >&2
      exit 1
    fi
    echo $defaultname
    return
  fi

  echo $curname
  return
}

satdir32=`setdefaultsatdir "$satdir32" "$SATDIR32" "32-bit solvers" "-3"`
if test $? = "1"; then exit 1; fi
satdir64=`setdefaultsatdir "$satdir64" "$SATDIR64" "64-bit solvers" "-3"`
if test $? = "1"; then exit 1; fi
satdir32compat=`setdefaultsatdir "$satdir32compat" "$SATDIR32" "32-bit compat solvers" "-3"`
if test $? = "1"; then exit 1; fi
satdir64compat=`setdefaultsatdir "$satdir64compat" "$SATDIR64" "64-bit compat solvers" "-3"`
if test $? = "1"; then exit 1; fi

if test "$satdir32" = "$satdir32compat"; then
  echo "NB: no compat-specific 32 bit solver dir"
fi
if test "$satdir64" = "$satdir64compat"; then
  echo "NB: no compat-specific 64 bit solver dir"
fi

# Tell the user about what version of Z3 we're about to compile with

function printz3 {
  z3ver=`$1/z3/bin/z3 -version | cut "--delim= " -f 3`
  echo "Z3 for $2: $z3ver"
}

printz3 $satdir64 "Linux64"
printz3 $satdir32 "Linux32"
printz3 $satdir64compat "LinuxCompat64"
printz3 $satdir32compat "LinuxCompat32"

# Find whatever the current head is
CURHEAD=`git symbolic-ref HEAD`

if test $? != 0; then
  # Not checked out a symbolic ref right now
  CURHEAD=`cat .git/HEAD`
fi

# Strip "refs/heads/" or suchlike from CURHEAD
CURHEAD=`basename $CURHEAD`

# Then, checkout whatever we've been told to release
# Allow the user to have a dirty tree, but bitch about it.
# Are there modified files?
git status -s | grep -v "^??" > /dev/null 2>&1
if test $? = "0"; then
  treeisdirty=1
  echo "***********************"
  echo "Your git tree is dirty:"
  git status -s
  echo "And it's going to get built into this release. Is that OK?"
  echo "Hit enter to continue; ctrl+c otherwise"
  read
  echo "Kay"
else
  treeisdirty=0
fi

if test -z "$targetrefname"; then
  switchedref=0
else
  switcheref=1
  git checkout $targetrefname > /dev/null
  if test $? != 0; then
    echo "Couldn't checkout $targetrefname"
    exit 1
  fi
fi

# And wrap all our modifications into a function, so that upon error we can
# cleanly remove all changes to the checked out copy.

function buildstep () {
  make clean > /dev/null 2>&1
  make > /dev/null 2>&1

  if test $? != 0; then
    echo "Build failed."
    return 1
  fi

  cp esbmc/esbmc $1
}

function dobuild () {

  # Install our configuration files.
  cp ./scripts/release_config.inc ./config.inc
  cp ./scripts/release_local.inc ./local.inc

  # And build build build
  rm -rf .release
  mkdir .release

  # For release builds, no debug information
  export EXTRACFLAGS="-DNDEBUG"
  export EXTRACXXFLAGS="-DNDEBUG"

  # Configure sat...
  export SATDIR32=$satdir32
  export SATDIR64=$satdir64

  # Override configuration in config.inc
  export EXTERN_ESBMC_CONFIG=1

  # And some actual config...
  export LINUX=1
  export TARGET64=1
  echo "Building 64 bit ESBMC"
  buildstep ".release/esbmc"
  if test $? != 0; then return $?; fi

  export STATICLINK=1
  echo "Building 64 bit static ESBMC"
  buildstep ".release/esbmc_static"
  if test $? != 0; then return $?; fi

  unset TARGET64
  unset STATICLINK
  export TARGET32=1
  echo "Building 32 bit ESBMC"
  buildstep ".release/esbmc32"
  if test $? != 0; then return $?; fi

  export STATICLINK=1
  echo "Building 32 bit static ESBMC"
  buildstep ".release/esbmc32_static"
  if test $? != 0; then return $?; fi

  unset TARGET32
  unset STATICLINK
  export TARGET64=1
  export LINUXCOMPAT=1
  export SATDIR32=$satdir32compat
  export SATDIR64=$satdir64compat
  echo "Building 64 bit compat ESBMC"
  buildstep ".release/esbmc_compat"
  if test $? != 0; then return $?; fi

  unset TARGET64
  export TARGET32=1
  echo "Building 32 bit compat ESBMC"
  buildstep ".release/esbmc32_compat"
  if test $? != 0; then return $?; fi

  unset LINUXCOMPAT
  unset TARGET32
  unset LINUX
  export WIN_MINGW32=1
  export TARGET64=1
  export SATDIR32=$satdir32
  export SATDIR64=$satdir64
  echo "Building 64 bit Windows binary"
  buildstep ".release/esbmc_windows"
  if test $? != 0; then return $?; fi

  unset TARGET64
  export TARGET32=1
  echo "Building 32 bit Windows binary"
  buildstep ".release/esbmc32_windows"
  if test $? != 0; then return $?; fi
}

function cleanup () {
  echo "Cleaning up"
  make clean > /dev/null 2>&1

  if test $switchedref = "1"; then
    # Check back out whatever ref we had before.
    git checkout $CURHEAD
  fi
}

function buildtgz {
  version=$1
  suffix=$2
  binpath=$3

  tmpdirname=`mktemp -d`
  projname="esbmc-$version-$suffix"
  dirname="$tmpdirname/$projname"
  mkdir $dirname
  mkdir $dirname/bin
  mkdir $dirname/licenses
  mkdir $dirname/smoke-tests

  # Copy data in
  cp scripts/README $dirname
  cp scripts/release-notes.txt $dirname
  cp $binpath $dirname/bin/esbmc
  cp scripts/licenses/* $dirname/licenses
  cp regression/smoke-tests/* $dirname/smoke-tests/

  # Create a tarball
  tar -czf .release/$projname.tgz -C $tmpdirname $projname
}

function buildtarballs() {
  version=$1

  buildtgz "$version" "linux-64" ".release/esbmc"
  buildtgz "$version" "linux-32" ".release/esbmc32"
  buildtgz "$version" "linux-64-static" ".release/esbmc_static"
  buildtgz "$version" "linux-32-static" ".release/esbmc32_static"
  buildtgz "$version" "linux-64-compat" ".release/esbmc_compat"
  buildtgz "$version" "linux-32-compat" ".release/esbmc32_compat"
  buildtgz "$version" "windows-64" ".release/esbmc_windows"
  buildtgz "$version" "windows-32" ".release/esbmc32_windows"
}

# If we get sigint/term/hup, cleanup before quitting.
trap "echo 'Exiting'; cleanup; exit 1" SIGHUP SIGINT SIGTERM

dobuild

# We now have a set of binaries (or an error)
if test $? != 0; then
  echo "Build failed"
fi

# If there's no target ref name, use current branch or hash.
if test -z $targetrefname; then
  buildtarballs $CURHEAD
else
  buildtarballs $targetrefname
fi

cleanup

# fini
