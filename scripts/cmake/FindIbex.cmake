# Module to find IBEX

if(DOWNLOAD_DEPENDENCIES AND (NOT DEFINED IBEX_DIR))
   # TODO: there might be a better way of doing this!
   if(ENABLE_WERROR)
      add_compile_options(-Wno-error)
   endif()
   include(CPM)
   CPMAddPackage(
      NAME ibex
      DOWNLOAD_ONLY YES
      URL https://github.com/ibex-team/ibex-lib/archive/refs/tags/ibex-2.9.1.tar.gz)

   message("[ibex] Setting up ibex")
   set(_LP_LIB none)
   if(ACADEMIC_BUILD)
      message(WARNING "[ibex] Academic build: LP solver SO-Plex required")
      set(_LP_LIB soplex)
   endif()

   execute_process(
      COMMAND cmake
      -DCMAKE_INSTALL_PREFIX=${ibex_BINARY_DIR}
      -DCMAKE_BUILD_TYPE=Release
      -DLP_LIB=${_LP_LIB}
      -B ${ibex_SOURCE_DIR}/build
      -S ${ibex_SOURCE_DIR})

   message("[ibex] Installing ibex")
   execute_process(
      COMMAND cmake --build . --target install --parallel 4
      WORKING_DIRECTORY ${ibex_SOURCE_DIR}/build)

   set(IBEX_DIR ${ibex_BINARY_DIR})

   if(ENABLE_WERROR)
      include(WError)
   endif()

endif()
find_package(PkgConfig)

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${IBEX_DIR}/share/pkgconfig")
pkg_check_modules(IBEX REQUIRED IMPORTED_TARGET ibex)
