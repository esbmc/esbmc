# This test don't work in Windows and neither macOS
if(UNIX AND NOT APPLE)
    new_unit_test(typecasttest "typecast.test.cpp" "clangcfrontend;bigint;util_esbmc;test_util_irep")
endif()
new_fuzz_test(typecastfuzz "typecast.fuzz.cpp" "clangcfrontend;bigint;util_esbmc;test_util_irep")