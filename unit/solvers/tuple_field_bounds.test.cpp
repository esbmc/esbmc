// Bounds contract for the tuple flatteners' field indices.
//
// smt_solver_baset::convert_member and the with_id case of convert_ast take a
// field index from the *expression's* struct type, while the AST they index
// carries the sort it was built from. Where a frontend leaves the two
// disagreeing, an unchecked index runs off the end of the flattener's vector:
// a read in project(), a write in tuple_node_smt_ast::update(), and in
// tuple_sym_smt_ast::update() a field match that never fires, dropping the
// write with no diagnostic at all.
//
// The mismatch cannot be built here. with2t::assert_consistency requires the
// update field to name a component of source_value->type and the with's own
// type to equal it (irep2_expr.cpp:352-363), and member2t asserts the same for
// its member (irep2_expr.h:1645) -- so well-formed IREP2 cannot express it, and
// a Debug build stops any attempt at the constructor. It is reachable only
// where those asserts are compiled out, which is where the guard has to hold.
// So this pins the guard itself; the end-to-end reproducers are the Solidity
// tests named in the PR.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>
#include <vector>
#include <irep2/irep2_utils.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <util/config/config.h>

namespace
{
type2tc struct_of(const std::vector<irep_idt> &names, const irep_idt &tag)
{
  std::vector<type2tc> members(names.size(), get_int32_type());
  return struct_type2tc(members, names, names, tag, false);
}
} // namespace

SCENARIO("a tuple field index is bounded by the tuple", "[solvers][tuple]")
{
  config.ansi_c.set_data_model(configt::LP64);

  const type2tc narrow = struct_of({"a", "b", "c"}, "Narrow");

  GIVEN("a tuple holding three fields")
  {
    THEN("every field it holds is accepted")
    {
      for (unsigned int idx = 0; idx < 3; idx++)
        REQUIRE_NOTHROW(check_tuple_field(idx, 3, narrow));
    }

    THEN("one past the end is rejected rather than indexed")
    {
      REQUIRE_THROWS_AS(check_tuple_field(3, 3, narrow), std::string);
    }

    THEN("an index from a wider type is rejected rather than indexed")
    {
      REQUIRE_THROWS_AS(check_tuple_field(7, 3, narrow), std::string);
    }

    THEN("an empty tuple admits no field")
    {
      REQUIRE_THROWS_AS(
        check_tuple_field(0, 0, struct_of({}, "Empty")), std::string);
    }
  }

  GIVEN("a union sort, which shares the flatteners' tuple lowering")
  {
    const type2tc u = union_type2tc(
      std::vector<type2tc>{get_int32_type()},
      std::vector<irep_idt>{"a"},
      std::vector<irep_idt>{"a"},
      "U");

    THEN("the bound holds there too, and naming it does not abort")
    {
      REQUIRE_NOTHROW(check_tuple_field(0, 1, u));
      REQUIRE_THROWS_AS(check_tuple_field(1, 1, u), std::string);
    }
  }
}
