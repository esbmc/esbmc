add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.txt
  COMMAND ${CMAKE_SOURCE_DIR}/src/scripts/buildidobj.sh ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS main.cpp esbmc_parseoptions.cpp bmc.cpp globals.cpp document_subgoals.cpp show_vcc.cpp options.cpp ansicfrontend cppfrontend clangcfrontend clangcppfrontend symex pointeranalysis langapi util_esbmc bigint solvers clibs # Depends on... everything else linked into esbmc. Add more as necessary.
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Mangling build ID object"
  VERBATIM
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c
  COMMAND sh -c "${CMAKE_SOURCE_DIR}/src/c2goto/headers/flail.sh buildidstring ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.txt > ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c"
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Flailing build ID object"
  VERBATIM
)

configure_file (
 "version.h.in"
 "${PROJECT_BINARY_DIR}/src/esbmc/version.h"
)

if(BUILD_STATIC)
    add_library(libesbmc esbmc_parseoptions.cpp bmc.cpp globals.cpp document_subgoals.cpp show_vcc.cpp options.cpp ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c)
else()
    add_library(libesbmc SHARED esbmc_parseoptions.cpp bmc.cpp globals.cpp document_subgoals.cpp show_vcc.cpp options.cpp ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c)
endif()

set_target_properties(libesbmc PROPERTIES VERSION "${ESBMC_VERSION}")
target_include_directories(libesbmc 
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_link_libraries(libesbmc ansicfrontend cppfrontend clangcfrontend clangcppfrontend symex pointeranalysis langapi util_esbmc bigint solvers clibs)
set_target_properties(libesbmc
  PROPERTIES OUTPUT_NAME esbmc)

add_executable (esbmc main.cpp)
target_include_directories(esbmc 
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_link_libraries(esbmc libesbmc)

if (DEFINED HAVE_PYTHON)
  add_library(esbmc_python python.cpp)
  target_link_libraries(esbmc esbmc_python ${PYTHON_LINK_LIBS})
  target_link_libraries(libesbmc esbmc_python ${PYTHON_LINK_LIBS})
endif()
