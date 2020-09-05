find_package(BISON)
find_package(FLEX)

BISON_TARGET(smtliby smtlib.ypp ${CMAKE_CURRENT_BINARY_DIR}/smtlib.cpp COMPILE_FLAGS "-d -psmtlib")
FLEX_TARGET(smtlibl smtlib_tok.lpp ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp COMPILE_FLAGS "--header-file=smtlib_tok.hpp -Psmtlib_tok -osmtlib_tok.cpp ${OS_FLEX_FLAGS}")
ADD_FLEX_BISON_DEPENDENCY(smtlibl smtliby)

# For Coverage
configure_file(smtlib.ypp  ${CMAKE_BINARY_DIR}/smtlib.ypp COPYONLY)
configure_file(smtlib_tok.lpp  ${CMAKE_BINARY_DIR}/smtlib_tok.lpp COPYONLY)

add_library(smtlib smtlib_conv.cpp ${CMAKE_CURRENT_BINARY_DIR}/smtlib.cpp ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
target_include_directories(smtlib     
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${Boost_INCLUDE_DIRS}
)