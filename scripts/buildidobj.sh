#!/bin/sh

if test $# != 1; then
  echo "An obj dir argument is required for buildobj.sh" >&2;
  exit 1;
fi

SFILE="$1/buildidobj.s"

echo -n "ESBMC built from " > $SFILE
echo -n `git rev-parse HEAD` >> $SFILE
echo -n " " >> $SFILE
echo -n `date` >> $SFILE
echo -n " by " >> $SFILE
echo -n `whoami` >> $SFILE
echo -n "@" >> $SFILE
echo -n `hostname` >> $SFILE
# Look for a dirty tree - any files that aren't '??' (i.e., untracked)
# or config.inc, which might differ for releases.
if test -z "`git status -s | grep -v "^??" | grep -v config.inc`"; then
  echo -n "" >> $SFILE;
else
  echo -n " (dirty tree)" >> $SFILE;
fi

OFILE="$1/buildiobj.o"

cd $1;
$LD $LDFLAGS -b binary -r ./buildidobj.s -o buildidobj.o
rm buildidobj.s
