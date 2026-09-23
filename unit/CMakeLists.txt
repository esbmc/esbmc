message(STATUS "Testing enabled")
include(TestConfiguration)

add_subdirectory(testing-utils)
add_subdirectory(goto-programs)
add_subdirectory(goto-symex)
add_subdirectory(big-int)
add_subdirectory(clang-c-frontend)
add_subdirectory(esbmc)

if(ENABLE_JIMPLE_FRONTEND)
 add_subdirectory(jimple-frontend)
endif()

if(ENABLE_PYTHON_FRONTEND)
  add_subdirectory(python-frontend)
endif()

if(ENABLE_LD_FRONTEND)
  add_subdirectory(ld-frontend)
endif()

add_subdirectory(langapi)
add_subdirectory(util)
add_subdirectory(c2goto)
add_subdirectory(irep2)
add_subdirectory(solvers)
add_subdirectory(pointer-analysis)

# scripts/irep2/bars.py measures the migration bars that docs/roadmap quotes,
# so a silent over-count there is the error reading the table cannot catch. Its
# self-test is python, hence add_test rather than a Catch2 subdirectory (#4715).
# Python is located the way regression/CMakeLists.txt does it: build.sh passes
# Python3_EXECUTABLE, and only WindowsConfiguration.cmake calls find_package.
if(DEFINED Python3_EXECUTABLE)
  set(_irep2_python ${Python3_EXECUTABLE})
else()
  find_package(Python COMPONENTS Interpreter)
  set(_irep2_python ${Python_EXECUTABLE})
endif()
if(_irep2_python)
  add_test(
    NAME irep2-bars-script
    COMMAND ${_irep2_python} ${CMAKE_SOURCE_DIR}/scripts/irep2/test_bars.py)
endif()
