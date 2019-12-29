# Module to install all licenses and text base files

# Assuming that this module is executed from <ROOT>/src/CMakeLists.txt

# LICENSES
install(FILES COPYING DESTINATION license)
install(DIRECTORY scripts/licenses/ DESTINATION license)

# EXTRA
install(FILES scripts/README DESTINATION .)
install(FILES scripts/release-notes.txt DESTINATION .)

# SHARE
install(DIRECTORY cpp/library/ DESTINATION share)