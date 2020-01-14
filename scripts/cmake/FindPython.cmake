# Find boost libraries for python

# TODO refactor this code
# Test for the presence of things required for python
if(ENABLE_PYTHON)
  
  include(FindPythonLibs)
  if ((NOT (DEFINED PYTHONLIBS_FOUND)) OR (NOT "${PYTHONLIBS_FOUND}"))
    set(NO_PYTHON 1)
    message(FATAL_ERROR "Didn't find python")
  else()
    if ("${PYTHONLIBS_VERSION_STRING}" VERSION_LESS 3.0.0)
      message(FATAL_ERROR "Found python ${PYTHONLIBS_VERSION_STRING}, but need at least python 3")
      set(NO_PYTHON 1)
    else()
      include_directories("${PYTHON_INCLUDE_DIRS}")
      #link_libraries("${PYTHON_LIBRARIES}")
      message(STATUS "Python version: ${PYTHONLIBS_VERSION_STRING}")
      string(REGEX REPLACE "^([0-9])\.[0-9]\.[0-9]$" "\\1" PYMAJOR ${PYTHONLIBS_VERSION_STRING})
      string(REGEX REPLACE "^[0-9]\.([0-9])\.[0-9]$"  "\\1" PYMINOR ${PYTHONLIBS_VERSION_STRING})
      if (("${PYMAJOR}" STREQUAL "") OR ("${PYMINOR}" STREQUAL ""))
        set (NO_PYTHON 1)
        message("Couldn't parse version string, disabling python")
      endif()
    endif()
  endif()

  find_library(BOOST_PYTHON "boost_python3")
  if ("${BOOST_PYTHON}" STREQUAL "BOOST_PYTHON-NOTFOUND")
    # Try just normal OS boost python
    message(FATAL_ERROR "error")
    find_library(BOOST_PYTHON "boost_python")
  endif()

  # And now try compiling it...
  try_compile (didbpcompile ${CMAKE_BINARY_DIR}
                ${CMAKE_SOURCE_DIR}/scripts/try_bp.cpp
                 CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${PYTHON_INCLUDE_DIRS} 
                 LINK_LIBRARIES ${BOOST_PYTHON} ${PYTHON_LIBRARIES} dl util expat z pthread
                 OUTPUT_VARIABLE DIDBPCOMPILE_MSG)

	    
  if ((NOT didbpcompile) AND DEBUG_TRY_RUNS)
    message("${DIDBPCOMPILE_MSG}")
  endif()

  if (didbpcompile)
    message(STATUS "Found boost python: ${BOOST_PYTHON}")
  else()
    set (NO_PYTHON 1)
    if ("${BOOST_PYTHON}" STREQUAL "BOOST_PYTHON-NOTFOUND")
      message("Couldn't find boost python")
    else()
      message("Found boost python but couldn't compile with it, it was at ${BOOST_PYTHON}")
    endif()
  endif()

  if (DEFINED NO_PYTHON)
    if (DEFINED ENABLE_PYTHON)
      message(SEND_ERROR "Python requested but couldn't find support")
    else()
      message("Python support disabled through lack of support")
    endif()
  else()
    message(STATUS "Enabling python support")
    add_definitions(-DWITH_PYTHON)
    include_directories(${PYTHON_INCLUDE_DIRS})
    set (HAVE_PYTHON "1")
    set (PYTHON_LINK_LIBS "${PYTHON_LIBRARIES};${BOOST_PYTHON}")
  endif()
else()
  set(ignoreme "${DISABLE_PYTHON}")
endif()
