# Module to find IBEX

if(DOWNLOAD_DEPENDENCIES AND (NOT DEFINED IBEX_DIR))   
   include(CPM)
   CPMAddPackage(
      NAME ibex
      DOWNLOAD_ONLY YES
      URL http://www.ibex-lib.org/ibex-2.8.9.tgz)   
      
   message("[ibex] Setting up ibex")
   if(ACADEMIC_BUILD)
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
endif()

find_package(PkgConfig)

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${IBEX_DIR}/share/pkgconfig")
pkg_check_modules(IBEX REQUIRED ibex)
