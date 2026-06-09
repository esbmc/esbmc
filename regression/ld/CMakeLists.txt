if(NOT ENABLE_LD_FRONTEND)
  return()
endif()

# Unit tests for each pipeline stage.
add_subdirectory(unit/parser)
add_subdirectory(unit/ir_gen)
add_subdirectory(unit/property)
