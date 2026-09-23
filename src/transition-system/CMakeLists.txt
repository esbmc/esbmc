add_library(transition-system transition_system.cpp ts_slice.cpp)
target_include_directories(transition-system
    PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(transition-system PUBLIC irep2 PRIVATE langapi fmt::fmt)

add_library(transition-system-extraction ts_extraction.cpp)
target_include_directories(transition-system-extraction
    PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(transition-system-extraction PUBLIC transition-system symex)
