#################################
# Modules                       #
#################################

add_subdirectory(util)
add_subdirectory(irep2)
add_subdirectory(goto-programs)
add_subdirectory(c2goto)

if(ENABLE_SOLIDITY_FRONTEND)
  add_subdirectory(solidity-frontend)
endif()

if(ENABLE_JIMPLE_FRONTEND)
  add_subdirectory(jimple-frontend)
endif()

if(ENABLE_PYTHON_FRONTEND)
  add_subdirectory(python-frontend)
endif()

if(ENABLE_LD_FRONTEND)
  add_subdirectory(ld-frontend)
endif()

add_subdirectory(clang-c-frontend)
add_subdirectory(pointer-analysis)
add_subdirectory(langapi)
add_subdirectory(big-int)
add_subdirectory(solvers)
add_subdirectory(clang-cpp-frontend)
add_subdirectory(goto-symex)
add_subdirectory(esbmc)

add_subdirectory(goto2c)

# ESBMC:: aliases for the libraries another project links against
foreach(esbmc_target
    util_esbmc irep2 bigint langapi filesystem string_pool usr_utils cache
    algorithms gotoprograms gotoalgorithms abstract-interpretation
    pointeranalysis symex solvers smt clibs goto2c esbmc-driver
    clangcfrontend clangcppfrontend)
  esbmc_add_alias(${esbmc_target})
endforeach()

set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS}" PARENT_SCOPE)
