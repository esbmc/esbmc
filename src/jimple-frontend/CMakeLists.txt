add_subdirectory(AST)
add_library(jimple
        jimple-language.cpp
        jimple-parser.cpp
        jimple-typecheck.cpp
        jimple-converter.cpp)

target_link_libraries(jimple
        PUBLIC esbmc_common jimple_ast
        PRIVATE clibs ESBMC::boost fmt::fmt nlohmann_json::nlohmann_json)
