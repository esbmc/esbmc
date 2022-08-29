
ls C:\vcpkg\installed\x64-windows\bin\
New-Item C:\Deps -itemtype directory
Copy-Item C:\vcpkg\installed\x64-windows\bin\boost_filesystem-vc143-mt-x64-1_80.dll C:\Deps
Copy-Item C:\vcpkg\installed\x64-windows\bin\boost_program_options-vc143-mt-x64-1_80.dll C:\Deps
