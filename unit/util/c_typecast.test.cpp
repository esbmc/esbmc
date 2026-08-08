// Implicit-conversion admission for the IREP2 overload of c_typecast.
//
// check_c_implicit_typecast has an irept and an expr2tc copy, written
// independently (see the note at the top of c_typecast.cpp). The IREP2 copy
// omitted `floatbv` entirely -- absent as a destination in every source branch
// and absent as a source branch of its own -- so it fell through to its final
// `return true` and rejected every implicit conversion involving a float.
// ESBMC represents a float as floatbv unless --fixedbv is given, so that
// rejected the default representation outright, and c_implicit_typecast became
// a silent no-op for its callers: python_adjust's assignment arm left an
// integer stored into a `double` lvalue
// (docs/roadmap/scope-relational-float-reconciliation.md §18.3). The same
// omission was fixed in get_c_type by esbmc/esbmc#6688.

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <util/lang/c_typecast.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>
#include <irep2/irep2_utils.h>

// get_c_type ranks an operand against config.ansi_c, which is zero-initialised
// bar int_128_width. Pin a model in main() rather than at namespace scope:
// `config` lives in another translation unit, so a static initialiser here
// would race its constructor.
int main(int argc, char *argv[])
{
  config.ansi_c.set_data_model(configt::LP64);
  return Catch::Session().run(argc, argv);
}

// check_c_implicit_typecast returns *false* to mean "permitted".
TEST_CASE(
  "check_c_implicit_typecast admits floatbv on both sides",
  "[c_typecast]")
{
  const type2tc i32 = get_int32_type();
  const type2tc dbl = double_type2();
  const type2tc flt = float_type2();

  SECTION("integer to floating point")
  {
    REQUIRE_FALSE(check_c_implicit_typecast(i32, dbl));
  }
  SECTION("Boolean to floating point")
  {
    REQUIRE_FALSE(check_c_implicit_typecast(get_bool_type(), dbl));
  }
  SECTION("floating point to integer")
  {
    REQUIRE_FALSE(check_c_implicit_typecast(dbl, i32));
  }
  SECTION("floating point to Boolean")
  {
    REQUIRE_FALSE(check_c_implicit_typecast(dbl, get_bool_type()));
  }
  SECTION("narrowing between floating-point widths")
  {
    REQUIRE_FALSE(check_c_implicit_typecast(dbl, flt));
  }
}

// The admission is not a blanket one: a conversion the irept overload also
// rejects must stay rejected, or the arms above would be untested by
// construction.
TEST_CASE(
  "check_c_implicit_typecast still rejects a struct source",
  "[c_typecast]")
{
  const type2tc st = struct_type2tc(
    std::vector<type2tc>{get_int32_type()},
    std::vector<irep_idt>{"f"},
    std::vector<irep_idt>{"f"},
    "tag-s");

  REQUIRE(check_c_implicit_typecast(st, double_type2()));
}

// The behaviour the callers depend on: the cast is actually inserted. A
// rejected conversion leaves the expression untouched, which is how an integer
// came to be stored into a `double` lvalue.
TEST_CASE("c_implicit_typecast converts an integer to double", "[c_typecast]")
{
  contextt ctx;
  namespacet ns(ctx);

  expr2tc e = gen_one(get_int32_type());
  REQUIRE_FALSE(c_implicit_typecast(e, double_type2(), ns));
  REQUIRE(is_floatbv_type(e->type));
}
