# Module to find IBEX

if(DOWNLOAD_DEPENDENCIES AND (NOT DEFINED IBEX_DIR))
   # TODO: there might be a better way of doing this!
   if(ENABLE_WERROR)
      add_compile_options(-Wno-error)
   endif()
   include(CPM)
   cpmaddpackage(
      NAME ibex
      DOWNLOAD_ONLY YES
      URL https://github.com/ibex-team/ibex-lib/archive/refs/tags/ibex-2.9.1.tar.gz)

   set(_ibex_build_dir "${ibex_BINARY_DIR}/build")
   set(_ibex_install_dir "${ibex_BINARY_DIR}/install")

   if(EXISTS "${_ibex_install_dir}/lib/libibex.a")
      message("[ibex] Found existing ibex installation, skipping build")
   else()
      # gaol's init() installs a process-wide round-toward-+infinity FPU mode
      # and never restores it, which is only safe if nothing else in the
      # process does floating-point. It is not: CaDiCaL's local-search
      # tabulates scores with `for (e = n; n; n = e * base)`, a loop that
      # terminates only when the sequence underflows to zero. Rounding upwards
      # pins it at DBL_TRUE_MIN, so the table grows until the process dies
      # (8 GB observed). Building gaol with GAOL_PRESERVE_ROUNDING makes each
      # interval operation save, set and restore the mode instead, which is
      # what its MinGW and MSVC configurations already do. The option is a
      # plain set() in gaol's CMakeLists, so -D cannot override it.
      set(_gaol_cmakelists
         "${ibex_SOURCE_DIR}/interval_lib_wrapper/gaol/3rd/gaol-4.2.3alpha0/CMakeLists.txt")
      if(EXISTS "${_gaol_cmakelists}")
         file(READ "${_gaol_cmakelists}" _gaol_contents)
         string(REPLACE "set (GAOL_PRESERVE_ROUNDING OFF)"
                        "set (GAOL_PRESERVE_ROUNDING ON)"
                        _gaol_patched "${_gaol_contents}")
         if(NOT _gaol_patched STREQUAL _gaol_contents)
            message("[ibex] Enabling GAOL_PRESERVE_ROUNDING")
            file(WRITE "${_gaol_cmakelists}" "${_gaol_patched}")
         endif()
      else()
         message(WARNING "[ibex] gaol CMakeLists not found at ${_gaol_cmakelists}; "
                         "GAOL_PRESERVE_ROUNDING not enabled")
      endif()

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
