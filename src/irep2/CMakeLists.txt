add_library(irep2 irep2.cpp irep2_type.cpp irep2_expr.cpp irep2_template_utils.cpp)
#set_target_properties(irep2 PROPERTIES COMPILE_FLAGS "-Wa,-mbig-obj")

target_include_directories(irep2 PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(irep2 PUBLIC fmt::fmt
  PUBLIC ${Boost_LIBRARIES}
  PRIVATE crypto_hash)

