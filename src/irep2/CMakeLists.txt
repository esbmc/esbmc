add_library(irep2
  irep2_type.cpp
  irep2_expr.cpp
  irep2_crc.cpp
  irep2_content_hash.cpp
  irep2_utils.cpp
  irep2_guard.cpp
  simplification_check.cpp
)

# irep2 and util_esbmc are mutually dependent: util_esbmc's headers are built on
# irep2, and irep2's objects call back into the util core (irept::remove_ref).
# Declaring only one direction leaves the other's references unresolved wherever
# nothing earlier on the link line happens to have pulled them in -- which is
# why the full binary linked while a unit test naming a subset did not. Stating
# the cycle lets CMake repeat the archives.
target_link_libraries(irep2 PUBLIC esbmc_common fmt::fmt bigint immer util_esbmc)
