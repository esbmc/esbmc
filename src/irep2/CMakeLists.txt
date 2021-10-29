add_library(irep2 irep2.cpp)
target_include_directories(irep2 PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(irep2 PUBLIC fmt::fmt
  PUBLIC ${Boost_LIBRARIES}
  PRIVATE crypto_hash)

