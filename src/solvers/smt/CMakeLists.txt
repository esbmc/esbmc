

add_library(smt
  smt_byteops.cpp
  smt_casts.cpp
  smt_conv.cpp
  smt_solver.cpp
  # IEEE754 constants/semantics and FP predicates extracted from smt_solver.cpp
  smt_fp_conv.cpp
  # IR-IEEE integer-encoding path (--int-encoding --ir-ieee)
  fp/ir_ieee_conv.cpp
  smt_memspace.cpp
  smt_overflow.cpp
  smt_bitcast.cpp
)

target_include_directories(smt
  PRIVATE ${Boost_INCLUDE_DIRS}
)

target_link_libraries(smt
  PUBLIC fmt::fmt
)
