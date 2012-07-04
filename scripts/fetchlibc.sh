#!/bin/sh

which curl > /dev/null 2>&1
if test $? != 0; then
  echo "You must have curl installed to download special-static-libc." >&2 
  exit 1
fi

which gpg > /dev/null 2>&1
if test $? != 0; then
  echo "You must have gpg installed to download special-static-libc." >&2 
  exit 1
fi

if test $# != 3; then
  echo "Usage: fetchlibc.sh infilename output_file path_to_keyring." >&2
  exit 1
fi

stat $1 > /dev/null 2>&1
if test $? == 0; then
  # It already exists.
  exit 0;
fi

tmplibc=`mktemp`
tmpsig=`mktemp`

echo "Fetching libc.a for linux 2.6.18"
curl -s --insecure https://jmorse.net/~jmorse/$1 -o $tmplibc
if test $? != 0; then
  echo "Failed to download libc.a." >&2
  exit 1
fi

echo "Fetching libc.a signature"
curl -s --insecure https://jmorse.net/~jmorse/$1.asc -o $tmpsig
if test $? != 0; then
  echo "Failed to download libc.a.asc." >&2
  exit 1
fi

echo "Verifying libc signature"
gpg --no-default-keyring --keyring $3 --verify $tmpsig $tmplibc >/dev/null 2>&1
if test $? != 0; then
  echo "Verification of libc file failed; bad signature." >&2
  exit 1
fi

echo "Success"
rm $tmpsig
mv $tmplibc $2
