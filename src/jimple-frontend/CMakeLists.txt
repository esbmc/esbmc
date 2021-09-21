add_subdirectory(AST)
add_library(jimple jimple-language.cpp jimple-parser.cpp)

# Boost is needed by anything that touches irep2
target_include_directories(jimple
        PUBLIC ${Boost_INCLUDE_DIRS})

target_link_libraries(jimple jimple_ast ${Boost_LIBRARIES} fmt::fmt nlohmann_json::nlohmann_json)