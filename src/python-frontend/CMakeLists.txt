add_library(pythonfrontend STATIC python_language.cpp python_converter.cpp)

target_include_directories(pythonfrontend
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pythonfrontend fmt::fmt nlohmann_json::nlohmann_json)