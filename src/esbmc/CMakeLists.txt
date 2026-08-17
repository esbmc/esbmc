
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

if(ENABLE_LD_FRONTEND)
  set(LD_FRONTEND_TARGETS ldfrontend)
endif()

if(ENABLE_GOTO_CONTRACTOR)
  set(GOTO_CONTRACTOR_TARGETS gotocontractor)
endif()

if(ENABLE_JIMPLE_FRONTEND)
  set(JIMPLE_FRONTEND_TARGETS jimple)
endif()

# A custom target always runs, so the version ID is re-derived on every build
# rather than once when the build tree is configured, which is what made
# --git-hash report the configure-time commit instead of the compiled one.
# buildidobj.py leaves the file untouched when only its timestamp would change,
# so the flail rule below, and the relink after it, still only fire when the ID
# actually changes.
add_custom_target(esbmc_buildid
  COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/buildidobj.py ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.txt
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating ESBMC version ID"
  VERBATIM
)
set(BUILD_ID_OBJ_OUT ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.txt)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c
  COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py -o ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c ${BUILD_ID_OBJ_OUT}
  DEPENDS  ${BUILD_ID_OBJ_OUT} ${CMAKE_SOURCE_DIR}/scripts/flail.py esbmc_buildid
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Flailing build ID object"
  VERBATIM
)

add_executable(esbmc main.cpp
    parseoptions/bmc_strategy.cpp
    parseoptions/command_line_options.cpp
    parseoptions/driver.cpp
    parseoptions/function_contracts.cpp
    parseoptions/goto_program.cpp
    parseoptions/k_induction.cpp
    parseoptions/process_goto_program.cpp
    parseoptions/property_monitors.cpp
    bmc.cpp property_report.cpp ranking_synthesis.cpp non_termination.cpp globals.cpp show_vcc.cpp options.cpp ${CMAKE_CURRENT_BINARY_DIR}/buildidobj.c)
target_include_directories(esbmc
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_link_libraries(esbmc ${OLD_FRONTEND_TARGETS} ${SOLIDITY_FRONTEND_TARGETS} ${PYTHON_FRONTEND_TARGETS}
                            ${LD_FRONTEND_TARGETS}
                            ${GOTO_CONTRACTOR_TARGETS} ${JIMPLE_FRONTEND_TARGETS}
                            clangcfrontend clangcppfrontend symex langapi
                            solvers clibs gotoalgorithms cache ${Boost_LIBRARIES} goto2c ${OS_INCLUDE_LIBS}
                            nlohmann_json::nlohmann_json)
if(MSVC)
  # Windows counterpart of the large stack main.cpp asks for elsewhere: its
  # default is 1MB, so deeply nested expressions overflow sooner still. The
  # reserve is address space, committed on demand.
  target_link_options(esbmc PRIVATE /STACK:536870912)
endif()
# mimalloc (when ENABLE_MIMALLOC resolved to a target): MI_OVERRIDE makes it
# replace the global allocator for the whole executable, speeding the
# allocation-heavy symex path. MIMALLOC_TARGET is empty if mimalloc was
# requested but unavailable.
if(MIMALLOC_TARGET STREQUAL "mimalloc-obj")
  # Fully-static build: add mimalloc's object file to the link AND allow
  # multiple definitions. glibc's static malloc.o defines malloc/free/realloc
  # as *strong* symbols (and gets force-pulled by other libs), so neither
  # archive ordering nor object-file preference avoids the clash — only
  # --allow-multiple-definition lets the link succeed, taking mimalloc's
  # definition (linked first) and dropping libc's. This is the standard
  # workaround for overriding malloc under -static glibc.
  target_sources(esbmc PRIVATE $<TARGET_OBJECTS:mimalloc-obj>)
  target_link_options(esbmc PRIVATE LINKER:--allow-multiple-definition)
elseif(MIMALLOC_TARGET)
  # Shared/dynamic libc: the archive override interposes fine. Link last so its
  # malloc/operator new override wins.
  target_link_libraries(esbmc ${MIMALLOC_TARGET})
endif()
install(TARGETS esbmc DESTINATION bin)
install(FILES ${CMAKE_SOURCE_DIR}/include/esbmc.h DESTINATION include)
