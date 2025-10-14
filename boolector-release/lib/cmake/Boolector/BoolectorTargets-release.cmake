#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Boolector::boolector" for configuration "Release"
set_property(TARGET Boolector::boolector APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Boolector::boolector PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libboolector.a"
  )

list(APPEND _cmake_import_check_targets Boolector::boolector )
list(APPEND _cmake_import_check_files_for_Boolector::boolector "${_IMPORT_PREFIX}/lib/libboolector.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
