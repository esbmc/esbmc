#!/bin/sh

if test ! -e esbmc; then
  echo "Please run from src/ dir";
  exit 1
fi

if test -z "$1"; then
  echo "ESBMC version number required on command line"
  exit 1
fi

tgzlist=`ls .release/*.tgz`

date=`date`
echo "Signatures for ESBMC binaries (http://esbmc.org)"
echo "Version $1 ($date)"
echo ""

# Generate checksums of each tgz
for file in $tgzlist;
do
  md5sum=`md5sum $file | cut "--delim= " -f 1`
  shasum=`shasum $file | cut "--delim= " -f 1`
  sha2sum=`sha256sum $file | cut "--delim= " -f 1`

  basename=`basename $file`
  echo "File $basename checksums:"
  echo "MD5:    $md5sum"
  echo "SHA:    $shasum"
  echo "SHA256: $sha2sum"
  echo ""
done

echo "Now run the following on this scripts \$output" >&2
echo "gpg -s --clearsign -u \$youremail < \$output > signedfile" >&2
