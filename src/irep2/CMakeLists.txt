add_library(irep2
  irep2_type.cpp
  irep2_expr.cpp
  irep2_crc.cpp
  irep2_utils.cpp
  irep2_guard.cpp
)

# irep2 headers include config.h -> cmdline.h -> boost/program_options.hpp,
# so irep2's sources need Boost's include path. This was previously inherited
# transitively from the (now-removed) crypto_hash target; declare it directly.
target_include_directories(irep2 PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(irep2 PUBLIC fmt::fmt bigint immer)
