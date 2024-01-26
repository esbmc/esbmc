add_library(langapi mode.cpp language_ui.cpp language_util.cpp)
target_include_directories(langapi
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_link_libraries(langapi PUBLIC fmt::fmt)
