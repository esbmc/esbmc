
# Download and extract prebuilt LLVM+Clang for Windows

$URL = "https://www.dropbox.com/s/0b6tw7uo5dx5p5h/llvm9d-win.zip?dl=1"
$LLVM_ZIP = "clang.zip"
$LLVM_DEST = "C:\deps\clang"

Invoke-WebRequest $URL -OutFile $LLVM_ZIP
Expand-Archive -LiteralPath $LLVM_ZIP -DestinationPath $LLVM_DEST
