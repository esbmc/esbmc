#!/usr/bin/env bash

set -x

if test $# != 1; then
  echo "Usage: scripts/export-pub.sh output.tgz" >&2
  exit 1
fi

# Procedure:
# 1) Export ESBMC to a tarball from git, avoiding local contamination
# 2) Extract, run autoconf
# 3) Delete private stuff (tests, 'papers')
# 4) Also this script
# 5) Tar back up as a release

tmpfile=`mktemp /tmp/esbmc_export_XXXXXX`
tmpdir=`mktemp -d /tmp/esbmc_export_XXXXXX`

function fin () {
  rm -rf $tmpdir
  if rm $tmpfile 2>/dev/null; then
    echo "" > /dev/null;
  fi
}

trap fin EXIT

rm $tmpfile

git archive -o $tmpfile HEAD
tar -xf $tmpfile -C $tmpdir
rm $tmpfile

# Ahem
mkdir $tmpdir/.git

sh -c "cd $tmpdir; scripts/autoboot.sh"

# Autobooted; delete private stuff

rm -rf $tmpdir/regression
rm -rf $tmpdir/papers
rm -rf $tmpdir/scripts/export-pub.sh

# Done. Tar back up.

workdir_parent=`dirname $tmpdir`
workdir_basename=`basename $tmpdir`
here=`pwd`
sh -c "cd $workdir_parent; tar -czf $here/$1 $workdir_basename"

# Everything else deleted when script exits
