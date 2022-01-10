new_unit_test(xmltest "xml.test.cpp" "util_esbmc;irep2;bigint")
new_unit_test(stdexprtest "std_expr.test.cpp" "util_esbmc;irep2;bigint")
new_unit_test(symboltest "symbol.test.cpp" "util_esbmc;irep2;bigint")
new_unit_test(string2integertest "string2integer.test.cpp" "util_esbmc;irep2;bigint")
new_unit_test(replace_symboltest "replace_symbol.test.cpp" "util_esbmc;irep2;bigint")
new_unit_test(ireptest "irep.test.cpp" "util_esbmc;irep2;bigint")
new_unit_test(filesystemtest "filesystem.test.cpp" "filesystem")
# Running the fuzzer normally would overflow the /tmp with files.
new_fast_fuzz_test(filesystemfuzz "filesystem.fuzz.cpp" "filesystem")
