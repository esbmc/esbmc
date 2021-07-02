add_library(message_handler message_handler.cpp)
add_library(message message.cpp)
add_library(default_message default_message.cpp)
add_library(message_stream message_stream.cpp)
add_library(fmt_message_handler fmt_message_handler.cpp)

target_include_directories(message_handler
        PRIVATE ${Boost_INCLUDE_DIRS}
)

target_include_directories(message
        PRIVATE ${Boost_INCLUDE_DIRS}
        )

target_link_libraries(message ${Boost_LIBRARIES})
target_link_libraries(fmt_message_handler fmt::fmt message_handler)
target_link_libraries(default_message message fmt_message_handler)
