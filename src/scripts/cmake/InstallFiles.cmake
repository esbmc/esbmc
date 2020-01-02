# Module to install all licenses and text base files

# Assuming that this module is executed from <ROOT>/src/CMakeLists.txt

set(CPACK_STRIP_FILES "bin/esbmc")
set(CPACK_SOURCE_STRIP_FILES "")
set(CPACK_PACKAGE_EXECUTABLES "bin/esbmc" "ESBMC BMC tool")
set(CPACK_PACKAGE_VERSION "${ESBMC_VERSION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${ESBMC_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${ESBMC_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${ESBMC_VERSION_PATCH}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/src/COPYING")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/src/scripts/README")

include(CPack)
# LICENSES
install(FILES "${CMAKE_SOURCE_DIR}/src/COPYING" DESTINATION license)
install(DIRECTORY scripts/licenses/ DESTINATION license)

# EXTRA
install(FILES scripts/README DESTINATION .)
install(FILES scripts/release-notes.txt DESTINATION .)