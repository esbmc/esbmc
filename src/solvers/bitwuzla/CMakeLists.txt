set(Bitwuzla_MIN_VERSION "0.1")

if(DEFINED Bitwuzla_DIR)
    set(ENABLE_BITWUZLA ON)
else()
    if(ENABLE_BITWUZLA AND DOWNLOAD_DEPENDENCIES)
        # See Boolector configurations for more details about this
        
        CPMAddPackage(
            NAME bitwuzla
            DOWNLOAD_ONLY YES
            GITHUB_REPOSITORY bitwuzla/bitwuzla
            GIT_TAG smtcomp-2021)

        message("[bitwuzla] Setting up boolector lingeling")
        execute_process(COMMAND "./contrib/setup-lingeling.sh"
            WORKING_DIRECTORY ${bitwuzla_SOURCE_DIR})
    
        message("[bitwuzla] Setting up btor2tools")
        execute_process(COMMAND "./contrib/setup-btor2tools.sh"
            WORKING_DIRECTORY ${bitwuzla_SOURCE_DIR})

        message("[bitwuzla] Setting up symfpu")
        execute_process(COMMAND "./contrib/setup-symfpu.sh"
            WORKING_DIRECTORY ${bitwuzla_SOURCE_DIR})
    
        message("[bitwuzla] Configuring build")
        execute_process(COMMAND ${bitwuzla_SOURCE_DIR}/configure.sh --prefix ${bitwuzla_BINARY_DIR}
            WORKING_DIRECTORY ${bitwuzla_SOURCE_DIR})
    
        message("[bitwuzla] Building...")
        execute_process(COMMAND make -j4 install
            WORKING_DIRECTORY ${bitwuzla_SOURCE_DIR}/build)
    
        set(Bitwuzla_DIR ${bitwuzla_BINARY_DIR})
    endif()

endif()

if(EXISTS $ENV{HOME}/bitwuzla)
    set(ENABLE_BITWUZLA ON)
endif()

if(ENABLE_BITWUZLA)
    find_package(Bitwuzla REQUIRED
            HINTS ${Bitwuzla_DIR}/lib/cmake $ENV{HOME}/bitwuzla)

    message(STATUS "Found Bitwuzla at: ${Bitwuzla_DIR}")
    message(STATUS "Bitwuzla version: ${Bitwuzla_VERSION}")
    if(Bitwuzla_VERSION VERSION_LESS Bitwuzla_MIN_VERSION)
        message(FATAL_ERROR "Expected version ${Bitwuzla_MIN_VERSION} or greater")
    endif()
   
    add_library(solverbitw bitwuzla_conv.cpp)
    target_include_directories(solverbitw
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(solverbitw fmt::fmt Bitwuzla::bitwuzla)

    target_link_libraries(solvers INTERFACE solverbitw)

    set(ESBMC_ENABLE_bitwuzla 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} bitwuzla" PARENT_SCOPE)
endif()
