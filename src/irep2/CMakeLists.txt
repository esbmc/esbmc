add_library(irep2 templates/irep2_templates_expr_data.cpp
  templates/irep2_templates_expr_ops.cpp
  templates/irep2_templates_expr_extra.cpp
  templates/irep2_templates_type.cpp
  templates/irep2_templates.cpp
  templates/irep2_template_utils.cpp
  irep2_type.cpp
  irep2_expr.cpp
)

target_include_directories(irep2 PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(irep2 PUBLIC fmt::fmt
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC bigint
  PRIVATE crypto_hash)
