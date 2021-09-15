add_library(jimple jimple-language.cpp)

# Boost is needed by anything that touches irep2
target_include_directories(jimple
        PUBLIC ${Boost_INCLUDE_DIRS}
        )

target_link_libraries(jimple ${Boost_LIBRARIES})