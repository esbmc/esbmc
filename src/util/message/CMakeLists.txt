add_library(message_handler message_handler.cpp)
add_library(message_stream message_stream.cpp)
add_library(fmt_message_handler fmt_message_handler.cpp)

target_link_libraries(fmt_message_handler fmt::fmt message_handler)
