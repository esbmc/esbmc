if(ENABLE_OLD_FRONTEND)
  set(BUILD_OBJ_OLD_TARGETS ansicfrontend cppfrontend)
else()
set(BUILD_OBJ_OLD_TARGETS "")
endif()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.txt
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/buildidobj.sh ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BUILD_OBJ_OLD_TARGETS} main.cpp esbmc_parseoptions.cpp bmc.cpp globals.cpp document_subgoals.cpp show_vcc.cpp options.cpp  clangcfrontend clangcppfrontend symex pointeranalysis langapi util_esbmc bigint solvers clibs # Depends on... everything else linked into esbmc. Add more as necessary.
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating ESBMC version ID"
  VERBATIM
)
set(BUILD_ID_OBJ_OUT ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.txt)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c
  COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py ${BUILD_ID_OBJ_OUT} ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c
  DEPENDS  ${BUILD_ID_OBJ_OUT} ${CMAKE_SOURCE_DIR}/scripts/flail.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Flailing build ID object"
  VERBATIM
)

add_executable (esbmc main.cpp esbmc_parseoptions.cpp bmc.cpp globals.cpp document_subgoals.cpp show_vcc.cpp options.cpp ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c)
target_include_directories(esbmc
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_link_libraries(esbmc ${BUILD_OBJ_OLD_TARGETS} clangcfrontend clangcppfrontend symex pointeranalysis langapi util_esbmc bigint solvers clibs)

install(TARGETS esbmc DESTINATION bin)
