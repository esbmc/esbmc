add_library(irep2 templates/irep2_templates.cpp
  templates/irep2_template_utils.cpp
  irep2_type.cpp
  irep2_expr.cpp
  irep2_utils.cpp
)

target_link_libraries(irep2 PUBLIC fmt::fmt bigint PRIVATE crypto_hash)

