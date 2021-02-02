set(Vampire_MIN_VERSION "5.1.0")

if(DEFINED Vampire_DIR)
    set(ENABLE_VAMPIRE ON)
endif()

if(EXISTS $ENV{HOME}/vampire_code_test)
#    set(ENABLE_VAMPIRE ON)
endif()

if(ENABLE_VAMPIRE)
    find_library(VAMPIRE_LIB vampire HINTS ${Vampire_DIR} $ENV{HOME}/vampire_code_test PATH_SUFFIXES lib bin)

    find_path(VAMPIRE_INCLUDE_DIRS1 Solver.hpp HINTS ${Vampire_DIR}/../Api $ENV{HOME}/vampire_code_test PATH_SUFFIXES include)


    if(VAMPIRE_INCLUDE_DIRS1 STREQUAL "VAMPIRE_INCLUDE_DIRS-NOTFOUND")
        message(FATAL_ERROR "Could not find vampire include headers, please check Vampire_DIR")
    endif()

    if(VAMPIRE_LIB STREQUAL "VAMPIRE_LIB-NOTFOUND")
        message(FATAL_ERROR "Could not find libvampire, please check Vampire_DIR")
    endif() 

    add_library(solvervamp vampire_conv.cpp)
    target_include_directories(solvervamp
            PRIVATE ${VAMPIRE_INCLUDE_DIRS1}
            PRIVATE ${VAMPIRE_INCLUDE_DIRS2}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if (BUILD_STATIC)
        target_link_libraries(solvervamp "${VAMPIRE_LIB}" -pthread "${LIBGOMP_LIB}" -ldl)
    else ()
        target_link_libraries(solvervamp "${VAMPIRE_LIB}")
    endif ()
    # Add to solver link
    target_link_libraries(solvers INTERFACE solvervamp)
    
    set(ESBMC_ENABLE_vampire 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} vampire" PARENT_SCOPE)
endif()
