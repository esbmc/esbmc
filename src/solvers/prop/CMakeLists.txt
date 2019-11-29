add_library(prop pointer_logic.cpp)
target_include_directories(prop     
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${Boost_INCLUDE_DIRS}
)
