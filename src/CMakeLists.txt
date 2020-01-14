#################################
# Modules                       #
#################################

include_directories(${CMAKE_SOURCE_DIR}/src)
add_subdirectory (goto-programs)
add_subdirectory (c2goto)
add_subdirectory (ansi-c)
add_subdirectory (clang-c-frontend)
add_subdirectory (cpp)
add_subdirectory (pointer-analysis)
add_subdirectory (langapi)
add_subdirectory (util)
add_subdirectory (big-int)
add_subdirectory (solvers)
add_subdirectory (clang-cpp-frontend)
add_subdirectory (goto-symex)
add_subdirectory (esbmc)
