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
      URL https://github.com/ibex-team/ibex-lib/archive/refs/tags/ibex-2.8.9.tar.gz)

   message("[ibex] Setting up ibex")
   if(ACADEMIC_BUILD)
      message(WARNING "the version of ibex you have is ZIB licensed, distribution is impossible.")
      execute_process(COMMAND ./waf --prefix=${ibex_BINARY_DIR} --lp-lib=soplex  configure
         WORKING_DIRECTORY ${ibex_SOURCE_DIR})
   else()
      execute_process(COMMAND ./waf --prefix=${ibex_BINARY_DIR} configure
         WORKING_DIRECTORY ${ibex_SOURCE_DIR})
   endif()
   message("[ibex] Installing ibex")
   execute_process(COMMAND ./waf -j4 install
      WORKING_DIRECTORY ${ibex_SOURCE_DIR})

   set(IBEX_DIR ${ibex_BINARY_DIR})

   if(ENABLE_WERROR)
      include(WError)
   endif()

endif()

find_package(PkgConfig)

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${IBEX_DIR}/share/pkgconfig")
pkg_search_module(IBEX REQUIRED IMPORTED_TARGET ibex)
pkg_check_modules(IBEX REQUIRED ibex)
