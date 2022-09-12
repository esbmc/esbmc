message(STATUS "Testing enabled")
include(TestConfiguration)

add_subdirectory(testing-utils)
add_subdirectory(goto-programs)
add_subdirectory(big-int)
add_subdirectory(clang-c-frontend)

if(ENABLE_JIMPLE_FRONTEND)
add_subdirectory (jimple-frontend)
endif()

add_subdirectory(util)
add_subdirectory(c2goto)
add_subdirectory(irep2)
