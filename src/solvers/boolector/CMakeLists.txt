set(Boolector_MIN_VERSION "3.2.0")
include(CPM)
include(ExecuteOrFail)

if(DEFINED Boolector_DIR)
  set(ENABLE_BOOLECTOR ON)
else()
  if(ENABLE_BOOLECTOR AND DOWNLOAD_DEPENDENCIES)
    # Sadly... Boolector requires some manual execution
    # before invoking its scripts. Thus, we can't use
    # FetchContent here
    cpmaddpackage(
      NAME boolector
      DOWNLOAD_ONLY YES
      GITHUB_REPOSITORY rafaelsamenezes/boolector
      GIT_TAG esbmc2)

    # We can't use ExternalProject because we need it ready
    # at configuration time

    # TODO: Maybe we should do a PR to boolector repo and make our
    # life easier in here.
    # The setup-*.sh scripts download a pinned tarball with a curl that has
    # neither -f nor --retry, so a transient HTTP error is unpacked as if it
    # were the archive ("gzip: stdin: not in gzip format"). Retry them.
    message("[boolector] Setting up lingeling")
    esbmc_execute_with_retry(
      TAG "boolector/lingeling"
      RETRIES 3
      COMMAND "./contrib/setup-lingeling.sh"
      WORKING_DIRECTORY ${boolector_SOURCE_DIR})

    message("[boolector] Setting up btor2tools")
    esbmc_execute_with_retry(
      TAG "boolector/btor2tools"
      RETRIES 3
      COMMAND "./contrib/setup-btor2tools.sh"
      WORKING_DIRECTORY ${boolector_SOURCE_DIR})

    message("[boolector] Configuring build")
    esbmc_execute_or_fail(
      TAG "boolector/configure"
      COMMAND ${boolector_SOURCE_DIR}/configure.sh --prefix ${boolector_BINARY_DIR}
      WORKING_DIRECTORY ${boolector_SOURCE_DIR})

    message("[boolector] Building...")
    esbmc_execute_or_fail(
      TAG "boolector/build"
      COMMAND make -j4 install
      WORKING_DIRECTORY ${boolector_SOURCE_DIR}/build)

    set(Boolector_DIR ${boolector_BINARY_DIR})
    endif()

endif()


if(ENABLE_BOOLECTOR)
  find_package(Boolector REQUIRED HINTS ${Boolector_DIR}/lib/cmake
               $ENV{HOME}/boolector)

  message(STATUS "Found Boolector at: ${Boolector_DIR}")
  message(STATUS "Boolector version: ${Boolector_VERSION}")
  if(Boolector_VERSION VERSION_LESS Boolector_MIN_VERSION)
    message(FATAL_ERROR "Expected version ${Boolector_MIN_VERSION} or greater")
  endif()

  add_library(solverbtor boolector_conv.cpp)
  target_include_directories(
    solverbtor
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(solverbtor Boolector::boolector fmt::fmt)

  target_link_libraries(solvers INTERFACE solverbtor)

  set(ESBMC_ENABLE_boolector 1 PARENT_SCOPE)
  set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} boolector"
      PARENT_SCOPE)
endif()
