#!/bin/bash -e

VER=1.24.99

if test ! -e .git; then
  echo "Please run from ESBMC root dir";
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
