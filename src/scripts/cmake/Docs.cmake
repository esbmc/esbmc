# Module to generate ESBMC docs

if(BUILD_DOC)
    find_package(Doxygen REQUIRED dot)

    doxygen_add_docs(docs
            ${PROJECT_SOURCE_DIR}/src
            ALL
            COMMENT "Generating API documentation with Doxygen")

    set(DOXYGEN_GENERATE_HTML NO)
    set(DOXYGEN_GENERATE_MAN YES)

    doxygen_add_docs(man
            ${PROJECT_SOURCE_DIR}/src
            ALL
            COMMENT "Generating API documentation with Doxygen")

    install(DIRECTORY ${PROJECT_BINARY_DIR}/src/man DESTINATION share)
    install(DIRECTORY ${PROJECT_BINARY_DIR}/src/html DESTINATION share)
endif()
