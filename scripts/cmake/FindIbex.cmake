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

   set(_ibex_build_dir "${ibex_BINARY_DIR}/build")
   set(_ibex_install_dir "${ibex_BINARY_DIR}/install")

   message("[ibex] Configuring ibex with CMake")
   set(_ibex_cmake_args
      -DCMAKE_INSTALL_PREFIX=${_ibex_install_dir}
      -DCMAKE_BUILD_TYPE=Release)
   if(ACADEMIC_BUILD)
      message(WARNING "the version of ibex you have is ZIB licensed, distribution is impossible.")
      list(APPEND _ibex_cmake_args -DLP_LIB=soplex)
   endif()
   execute_process(
      COMMAND ${CMAKE_COMMAND} -S ${ibex_SOURCE_DIR} -B ${_ibex_build_dir} ${_ibex_cmake_args}
      RESULT_VARIABLE _ibex_configure_result)
   if(NOT _ibex_configure_result EQUAL 0)
      message(FATAL_ERROR "[ibex] CMake configure failed")
   endif()

   message("[ibex] Building ibex")
   execute_process(
      COMMAND ${CMAKE_COMMAND} --build ${_ibex_build_dir} --parallel 4
      RESULT_VARIABLE _ibex_build_result)
   if(NOT _ibex_build_result EQUAL 0)
      message(FATAL_ERROR "[ibex] Build failed")
   endif()

   message("[ibex] Installing ibex")
   execute_process(
      COMMAND ${CMAKE_COMMAND} --install ${_ibex_build_dir}
      RESULT_VARIABLE _ibex_install_result)
   if(NOT _ibex_install_result EQUAL 0)
      message(FATAL_ERROR "[ibex] Install failed")
   endif()

   set(IBEX_DIR ${_ibex_install_dir})

   if(ENABLE_WERROR)
      include(WError)
   endif()

endif()

find_package(PkgConfig)

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${IBEX_DIR}/share/pkgconfig:${IBEX_DIR}/lib/pkgconfig")
pkg_search_module(IBEX REQUIRED IMPORTED_TARGET ibex)
pkg_check_modules(IBEX REQUIRED ibex)
