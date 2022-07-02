# Download and extract prebuilt LLVM+Clang for Windows

$URL = "https://sourceforge.net/projects/esbmc-deps/files/clang.zip/download"
$LLVM_ZIP = "clang.zip"
$LLVM_DEST = "C:\deps\clang"

Invoke-WebRequest $URL -OutFile $LLVM_ZIP -UserAgent "NativeHost"
Expand-Archive -LiteralPath $LLVM_ZIP -DestinationPath $LLVM_DEST

$Z3 = "https://sourceforge.net/projects/esbmc-deps/files/z3.zip/download"
$Z3_ZIP = "z3.zip"
$Z3_DEST = "C:\deps"

Invoke-WebRequest $Z3 -OutFile $Z3_ZIP -UserAgent "NativeHost"
Expand-Archive -LiteralPath $Z3_ZIP -DestinationPath $Z3_DEST

ls C:\vcpkg\installed\x64-windows\bin\
Copy-Item C:\vcpkg\installed\x64-windows\bin\boost_filesystem-vc143-mt-x64-1_79.dll C:\Deps
Copy-Item C:\vcpkg\installed\x64-windows\bin\boost_program_options-vc143-mt-x64-1_79.dll C:\Deps
