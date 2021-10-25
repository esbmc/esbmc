new_unit_test(jimple_asttest "jimple_ast.test.cpp" "clibs;jimple_ast;${Boost_LIBRARIES};util_esbmc;bigint;nlohmann_json::nlohmann_json")

#clibs jimple_ast ${Boost_LIBRARIES} fmt::fmt nlohmann_json::nlohmann_json
