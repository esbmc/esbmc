# For some reason those tests are causing issues in Windows and macos (compilation)
if(UNIX AND NOT APPLE)
    new_unit_test(xmltest "xml.test.cpp" "util_esbmc")
    new_unit_test(stdexprtest "std_expr.test.cpp" "util_esbmc;bigint")
    new_unit_test(symboltest "symbol.test.cpp" "util_esbmc;bigint")
    new_unit_test(string2integertest "string2integer.test.cpp" "util_esbmc;bigint")
    new_unit_test(replace_symboltest "replace_symbol.test.cpp" "util_esbmc;bigint")
    new_unit_test(ireptest "irep.test.cpp" "util_esbmc;bigint")
endif()