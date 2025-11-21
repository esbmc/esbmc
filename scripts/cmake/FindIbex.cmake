# Module to find IBEX

if(DOWNLOAD_DEPENDENCIES AND (NOT DEFINED IBEX_DIR))
  if(ENABLE_WERROR)
    add_compile_options(-Wno-error)
  endif()
  include(CPM)
  if(ACADEMIC_BUILD)
    cpmaddpackage(
      NAME
      ibex
      DOWNLOAD_ONLY
      NO
      URL
      https://github.com/ibex-team/ibex-lib/archive/refs/tags/ibex-2.9.1.tar.gz
      OPTIONS
      "CMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/ibex_install"
      "LP_LIB=soplex")
  else()
    cpmaddpackage(
      NAME
      ibex
      DOWNLOAD_ONLY
      NO
      URL
      https://github.com/ibex-team/ibex-lib/archive/refs/tags/ibex-2.9.1.tar.gz
      OPTIONS
      "CMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/ibex_install")
  endif()
  set(IBEX_TARGET "ibex")
  find_package(ibex REQUIRED)
else()
  if(DEFINED IBEX_DIR)
    find_package(ibex REQUIRED PATHS ${IBEX_DIR} NO_DEFAULT_PATH)
  else()
    find_package(ibex REQUIRED)
  endif()
  set(IBEX_TARGET "Ibex::ibex")
endif()
