# Module to generate ESBMC docs.
#
# This drives the top-level .doxygen Doxyfile so that `ninja docs`, the CI
# workflow, and a local `doxygen .doxygen` all share a single configuration.
# Enable with -DBUILD_DOC=On (requires doxygen and graphviz). Output is written
# to docs/html in the source tree (gitignored).

if(BUILD_DOC)
    find_package(Doxygen REQUIRED dot)

    add_custom_target(docs ALL
        COMMAND ${PROJECT_SOURCE_DIR}/scripts/gen-docs.sh
        COMMENT "Generating API documentation from .doxygen")

    install(DIRECTORY ${PROJECT_SOURCE_DIR}/docs/html DESTINATION share/doc/esbmc)
endif()
