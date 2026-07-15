// H-A8: get_sub_expr index accumulation over the real exprs. The generic walk
// (do_get_sub_expr in irep2_utils.cpp) accumulates an offset `it` across
// fields and, on a std::vector field, returns &item[idx - it]. This checks
// that indexing returns the *correct* element (not just non-null), including
// the case where the vector field is reached with it > 0.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>

namespace
{
uint64_t sub_val(const expr2tc &e, size_t i)
{
  const expr2tc *p = e->get_sub_expr(i);
  REQUIRE(p != nullptr);
  return to_constant_int2t(*p).value.to_uint64();
}
} // namespace

TEST_CASE("get_sub_expr returns the correct vector element", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  type2tc arr_ty = array_type2tc(get_uint_type(32), expr2tc(), true);
  std::vector<expr2tc> members;
  for (unsigned i = 0; i < 5; ++i)
    members.push_back(gen_ulong(10 + i));
  expr2tc arr = constant_array2tc(arr_ty, members);

  REQUIRE(arr->get_num_sub_exprs() == 5);
  for (size_t i = 0; i < 5; ++i)
    REQUIRE(sub_val(arr, i) == 10 + i);
  REQUIRE(arr->get_sub_expr(5) == nullptr);
}

// code_function_call2t fields are (type, ret, function, operands-vector), so
// the operands vector is reached with the accumulator it == 2 — exercising
// the idx - it arithmetic with a non-zero base.
TEST_CASE(
  "get_sub_expr indexes across scalar and vector fields",
  "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  std::vector<expr2tc> args{gen_ulong(10), gen_ulong(11), gen_ulong(12)};
  expr2tc call = code_function_call2tc(gen_ulong(100), gen_ulong(200), args);

  REQUIRE(call->get_num_sub_exprs() == 5); // ret, function, 3 operands
  REQUIRE(sub_val(call, 0) == 100);        // ret
  REQUIRE(sub_val(call, 1) == 200);        // function
  REQUIRE(sub_val(call, 2) == 10);         // operands[0] (it == 2)
  REQUIRE(sub_val(call, 3) == 11);
  REQUIRE(sub_val(call, 4) == 12);
  REQUIRE(call->get_sub_expr(5) == nullptr);
}
