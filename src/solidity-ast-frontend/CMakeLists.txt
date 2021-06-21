add_subdirectory(libUtils)
add_subdirectory(libSif)

add_library(solidityfrontend_stuff solidity_ast_language.cpp typecast.cpp solidity_convert.cpp solidity_type.cpp
  solidity_decl_tracker.cpp solidity_convert_literals.cpp)
target_include_directories(solidityfrontend_stuff
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CLANG_INCLUDE_DIRS}
    PRIVATE ${Boost_INCLUDE_DIRS}
)
#target_link_libraries(solidityfrontend_stuff gotoprograms clangTooling nlohmann_json::nlohmann_json clangcfrontend_stuff)
target_link_libraries(solidityfrontend_stuff gotoprograms nlohmann_json::nlohmann_json clangcfrontend_stuff)

add_library(solidityfrontend INTERFACE)
target_link_libraries(solidityfrontend INTERFACE solidityfrontend_stuff solidityutil soliditysif)
