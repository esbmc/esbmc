Write-Host "Set TLS1.2"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor "Tls12"
choco install -y nsis.portable --ignore-checksums &&
./scripts/winflexbison_install.ps1

od.exe --version &&

vcpkg.exe integrate install &&
vcpkg.exe install boost-filesystem:x64-windows-static boost-multiprecision:x64-windows-static boost-date-time:x64-windows-static boost-test:x64-windows-static boost-multi-index:x64-windows-static boost-crc:x64-windows-static boost-property-tree:x64-windows-static boost-process:x64-windows-static boost-uuid:x64-windows-static boost-program-options:x64-windows-static boost-iostreams:x64-windows-static &&

mkdir build &&
cd build &&
cmake.exe .. -A x64 -DVCPKG_TARGET_TRIPLET=x64-windows-static -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake -DENABLE_REGRESSION=On -DBUILD_TESTING=On -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX:PATH=C:/deps/esbmc -DENABLE_PYTHON_FRONTEND=On -DENABLE_SOLIDITY_FRONTEND=On -DENABLE_JIMPLE_FRONTEND=On -DDOWNLOAD_DEPENDENCIES=On -DENABLE_Z3=ON -DENABLE_SMTLIB=OFF &&
cmake --build . --target INSTALL  --config RelWithDebInfo
