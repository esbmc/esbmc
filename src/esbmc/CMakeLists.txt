if(ENABLE_OLD_FRONTEND)
  set(OLD_FRONTEND_TARGETS ansicfrontend cppfrontend)
endif()

if(ENABLE_SOLIDITY_FRONTEND)
  # we have set minimium LLVM version to 11
  if(LLVM_VERSION_MAJOR STREQUAL "14" OR LLVM_VERSION_MAJOR STREQUAL "15")
    message(FATAL_ERROR 
        "Unsupported LLVM version (${LLVM_VERSION_MAJOR}). 
         LLVM 14 and 15 have incomplete or unstable support for ExtInt and BitInt features, 
         which are required for the Solidity frontend to function properly. 
         Please use LLVM 11-13 for stable ExtInt support or LLVM 16+ for stable BitInt support.")
  endif()
  set(SOLIDITY_FRONTEND_TARGETS solidityfrontend)
endif()

if(ENABLE_PYTHON_FRONTEND)
  set(PYTHON_FRONTEND_TARGETS pythonfrontend)
endif()

if(ENABLE_GOTO_CONTRACTOR)
  set(GOTO_CONTRACTOR_TARGETS gotocontractor)
endif()

if(ENABLE_JIMPLE_FRONTEND)
  set(JIMPLE_FRONTEND_TARGETS jimple)
endif()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.txt
  COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/buildidobj.py ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating ESBMC version ID"
  VERBATIM
)
set(BUILD_ID_OBJ_OUT ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.txt)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c
  COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py -o ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c ${BUILD_ID_OBJ_OUT}
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
target_compile_features(esbmc PUBLIC cxx_std_17)
cmake_path(SET MY_MAGIC_PATH NORMALIZE "${cvc5_DIR}/../../")
cmake_path(ABSOLUTE_PATH MY_MAGIC_PATH OUTPUT_VARIABLE CVC_LINK_DIR)
message("Looking at ${CVC_LINK_DIR}")
target_link_libraries(esbmc PUBLIC ${OLD_FRONTEND_TARGETS} ${SOLIDITY_FRONTEND_TARGETS} ${PYTHON_FRONTEND_TARGETS}
                            ${GOTO_CONTRACTOR_TARGETS} ${JIMPLE_FRONTEND_TARGETS} ${JIMPLE_FRONTEND_TARGETS}
                            clangcfrontend clangcppfrontend filesystem symex pointeranalysis langapi util_esbmc bigint
                            solvers clibs gotoalgorithms cache ${Boost_LIBRARIES} goto2c ${OS_INCLUDE_LIBS})
target_link_directories(esbmc PUBLIC "${CVC_LINK_DIR}")
install(TARGETS esbmc DESTINATION bin)
