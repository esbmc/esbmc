# Download and extract prebuilt LLVM+Clang for Windows

$URL = "https://www.dropbox.com/s/z1gyschfa46yj6e/clang.zip?dl=1"
$LLVM_ZIP = "clang.zip"
$LLVM_DEST = "C:\deps\clang"

Invoke-WebRequest $URL -OutFile $LLVM_ZIP
Expand-Archive -LiteralPath $LLVM_ZIP -DestinationPath $LLVM_DEST

$Z3 = "https://github.com/Z3Prover/z3/releases/download/z3-4.8.9/z3-4.8.9-x64-win.zip"
$Z3_ZIP = "z3.zip"
$Z3_DEST = "C:\deps"

Invoke-WebRequest $Z3 -OutFile $Z3_ZIP
Expand-Archive -LiteralPath $Z3_ZIP -DestinationPath $Z3_DEST

$BOOST = "https://www.dropbox.com/s/n6846t9q9wxmzvi/boost-dll.zip?dl=1"
$BOOST_ZIP = "boost-dll.zip"
$BOOST_DEST = "C:\deps"

Invoke-WebRequest $BOOST -OutFile $BOOST_ZIP
Expand-Archive -LiteralPath $BOOST_ZIP -DestinationPath $BOOST_DEST
