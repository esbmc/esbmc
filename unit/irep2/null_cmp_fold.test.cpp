// The null-comparison fold decides `&sym == NULL` but must refuse an
// address_of over a dereference: `&*p` is p and may be NULL. Pinned
// here because no C input reaches the fold with that shape — clang
// folds `&*p` to `p` (C99 6.5.3.2p3) and `&p->m` lowers to pointer
// arithmetic over the refused base symbol; the shape is internal.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>

namespace
{
type2tc int_ptr()
{
  return pointer_type2tc(get_int32_type());
}

expr2tc null_ptr()
{
  return symbol2tc(int_ptr(), irep_idt("NULL"));
}
} // namespace

TEST_CASE("&obj == NULL folds to false", "[irep2][simplify]")
{
  expr2tc obj = symbol2tc(get_int32_type(), irep_idt("obj"));
  expr2tc eq = equality2tc(address_of2tc(int_ptr(), obj), null_ptr());
  expr2tc s = eq->simplify();
  REQUIRE(is_constant_bool2t(s));
  REQUIRE(!to_constant_bool2t(s).value);
}

TEST_CASE("&*p == NULL must not fold", "[irep2][simplify]")
{
  expr2tc p = symbol2tc(int_ptr(), irep_idt("p"));
  expr2tc roundtrip =
    address_of2tc(int_ptr(), dereference2tc(get_int32_type(), p));
  expr2tc eq = equality2tc(roundtrip, null_ptr());
  expr2tc s = eq->simplify();
  if (!is_nil_expr(s))
    REQUIRE(!is_constant_bool2t(s));
}
