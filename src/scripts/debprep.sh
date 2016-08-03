#!/bin/bash -e

if test $# != 1; then
  echo "Usage: scripts/debprep.sh version-num" >&2
  exit 1
fi

VER=$1

if test ! -e esbmc; then
  echo "Please run from src/ dir";
  exit 1
fi

tmpfile=`mktemp /tmp/esbmc_build_XXXXXX`
tmpdir=`mktemp -d /tmp/esbmc_build_XXXXXX`

function fin () {
  rm -rf $tmpdir
  if rm $tmpfile 2>/dev/null; then
    echo "" > /dev/null;
  fi
}

trap fin EXIT

rm $tmpfile
TARDIR=$tmpdir/esbmc-${VER}
mkdir $TARDIR

git archive -o $tmpfile HEAD
tar -xf $tmpfile -C $TARDIR
rm $tmpfile
cp -r scripts/debian $TARDIR

# Ahem
mkdir $TARDIR/.git

sh -c "cd $TARDIR; scripts/autoboot.sh"
here=`pwd`
sh -c "cd $tmpdir; tar -cjf $here/esbmc_${VER}.orig.tar.bz2 esbmc-${VER}"
