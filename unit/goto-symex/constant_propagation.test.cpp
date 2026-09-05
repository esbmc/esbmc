/*******************************************************************
 Module: goto_symex_statet::constant_propagation on immutable leaves

 Propagation is decided per symbol, but a member write is lowered to
 `obj = with(obj, "f", v)`, so whether an object keeps a propagated value is
 decided by the whole `with` chain. Before #7597 one symbolically-valued
 member write dropped the object, taking a sibling loop counter's value with
 it, and the loop guard never folded.

 The regression tests in regression/esbmc/github_7597* pin the end-to-end
 verdict. These pin the individual arms of the decision, which an end-to-end
 test cannot separate: which leaf shapes count as immutable, and which of the
 struct / array / literal paths carry one.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <cstring>
#include <string>
#include <vector>

#include <goto-symex/reachability_tree.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/lang/c_types.h>
#include <util/symtab/namespace.h>

#include "../testing-utils/goto_factory.h"

namespace
{
/** Owns everything a real execution state needs to stay alive. */
class engine
{
public:
  engine()
    : source("int main(void) { int x = 0; return x; }"),
      prog(goto_factory::get_goto_functions(
        source,
        goto_factory::Architecture::BIT_64)),
      ns(prog.context),
      opts(goto_factory::get_default_options(
        goto_factory::get_default_cmdline("test.c"))),
      rt(
        prog.functions,
        ns,
        opts,
        std::make_shared<symex_target_equationt>(ns),
        prog.context)
  {
    rt.setup_for_new_explore();
  }

  const goto_symex_statet &state()
  {
    return rt.get_cur_state().get_active_state();
  }

private:
  std::string source;
  program prog;
  namespacet ns;
  optionst opts;
  reachability_treet rt;
};

expr2tc
symbol_at(const type2tc &type, const char *name, symbol_renaming_level lev)
{
  return symbol2tc(type, irep_idt(name), lev, 0, 1, 0, 0);
}

expr2tc l2_int(const char *name = "c:test.c@F@main@v")
{
  return symbol_at(int_type2(), name, symbol_renaming_level::level2);
}

/** A nondet$ free variable: never assigned, so no L2 generation is minted. */
expr2tc nondet_int_symbol()
{
  return symbol_at(
    int_type2(), "nondet$symex::free_input", symbol_renaming_level::level0);
}

/** `struct { int i; int r; }`, the shape of #7597's counter and its sibling. */
type2tc pair_struct()
{
  std::vector<type2tc> members{int_type2(), int_type2()};
  std::vector<irep_idt> names{"i", "r"};
  return struct_type2tc(members, names, names, "pair");
}

/** `struct { pair inner; }`, to build a nested member read. */
type2tc nest_struct()
{
  std::vector<type2tc> members{pair_struct()};
  std::vector<irep_idt> names{"inner"};
  return struct_type2tc(members, names, names, "nest");
}

type2tc int_array(unsigned n)
{
  return array_type2tc(
    int_type2(), constant_int2tc(size_type2(), BigInt(n)), false);
}

expr2tc int_const(int v)
{
  return constant_int2tc(int_type2(), BigInt(v));
}

/** `member(source, name)` at @p type. */
expr2tc member_of(
  const expr2tc &source,
  const char *name,
  const type2tc &type = int_type2())
{
  return member2tc(type, source, irep_idt(name));
}

/** `source[idx]` at @p type. */
expr2tc index_of(
  const expr2tc &source,
  const expr2tc &idx,
  const type2tc &type = int_type2())
{
  return index2tc(type, source, idx);
}

/** `with(source, field, value)`, the lowering of a member/element write. */
expr2tc
with_field(const expr2tc &source, const char *field, const expr2tc &value)
{
  const type2tc str_type =
    array_type2tc(get_uint8_type(), gen_ulong(strlen(field) + 1), false);
  return with2tc(
    source->type,
    source,
    constant_string2tc(str_type, field, constant_string_kindt::DEFAULT),
    value);
}

expr2tc
with_index(const expr2tc &source, const expr2tc &idx, const expr2tc &value)
{
  return with2tc(source->type, source, idx, value);
}
} // namespace

TEST_CASE(
  "a member read out of an assigned-once object is propagatable",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc obj = symbol_at(
    pair_struct(), "c:test.c@F@main@IO", symbol_renaming_level::level2);

  // The value #7597's loop body writes: IO.in, read out of an L2 generation.
  REQUIRE(e.state().constant_propagation(with_field(
    symbol_at(
      pair_struct(), "c:test.c@F@main@VAR", symbol_renaming_level::level2),
    "r",
    member_of(obj, "i"))));
}

TEST_CASE(
  "an immutable leaf reaches through nested member and index reads",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc base = symbol_at(
    pair_struct(), "c:test.c@F@main@VAR", symbol_renaming_level::level2);
  const expr2tc arr =
    symbol_at(int_array(4), "c:test.c@F@main@A", symbol_renaming_level::level2);
  const expr2tc nest = symbol_at(
    nest_struct(), "c:test.c@F@main@N", symbol_renaming_level::level2);
  const expr2tc grid = symbol_at(
    array_type2tc(int_array(4), gen_ulong(2), false),
    "c:test.c@F@main@G",
    symbol_renaming_level::level2);

  // Each shape the peel loop accepts, as the update value of a struct chain.
  const expr2tc leaves[] = {
    l2_int(),
    nondet_int_symbol(),
    member_of(base, "i"),
    member_of(member_of(nest, "inner", pair_struct()), "i"),
    index_of(arr, int_const(2)),
    index_of(index_of(grid, int_const(1), int_array(4)), int_const(0)),
    typecast2tc(int_type2(), member_of(base, "i")),
  };

  for (const expr2tc &leaf : leaves)
  {
    CAPTURE(get_expr_id(leaf));
    REQUIRE(e.state().constant_propagation(with_field(base, "r", leaf)));
  }
}

TEST_CASE(
  "a leaf that is not assigned-once is not propagatable",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc base = symbol_at(
    pair_struct(), "c:test.c@F@main@VAR", symbol_renaming_level::level2);
  const expr2tc arr =
    symbol_at(int_array(4), "c:test.c@F@main@A", symbol_renaming_level::level2);

  // An L1 symbol still has generations to come, so a read of it is not fixed.
  const expr2tc l1 =
    symbol_at(int_type2(), "c:test.c@F@main@w", symbol_renaming_level::level1);
  const expr2tc l1_struct = symbol_at(
    pair_struct(), "c:test.c@F@main@P", symbol_renaming_level::level1);
  // A symbolic index names a different element on each evaluation.
  const expr2tc symbolic_index = index_of(arr, l2_int());
  // add2t is neither a read nor a propagatable constant here.
  const expr2tc arith = add2tc(int_type2(), l2_int(), l1);

  for (const expr2tc &leaf :
       {l1, member_of(l1_struct, "i"), symbolic_index, arith})
  {
    CAPTURE(get_expr_id(leaf));
    REQUIRE_FALSE(e.state().constant_propagation(with_field(base, "r", leaf)));
  }
}

TEST_CASE(
  "an array chain carries an immutable element",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc arr =
    symbol_at(int_array(4), "c:test.c@F@main@A", symbol_renaming_level::level2);

  // #7597's array half: A[1] written a symbolic value must not drop A[0].
  REQUIRE(e.state().constant_propagation(
    with_index(arr, int_const(1), nondet_int_symbol())));

  REQUIRE_FALSE(e.state().constant_propagation(with_index(
    arr,
    int_const(1),
    symbol_at(
      int_type2(), "c:test.c@F@main@w", symbol_renaming_level::level1))));
}

TEST_CASE(
  "an aggregate literal carries an immutable element",
  "[symex][constant-propagation]")
{
  engine e;

  // The union path already allowed this (#7446); struct and array literals
  // are the arms #7597 opened.
  std::vector<expr2tc> pair{int_const(1), nondet_int_symbol()};
  REQUIRE(
    e.state().constant_propagation(constant_struct2tc(pair_struct(), pair)));

  std::vector<expr2tc> elems{int_const(0), l2_int()};
  REQUIRE(
    e.state().constant_propagation(constant_array2tc(int_array(2), elems)));

  std::vector<expr2tc> mutable_elems{
    int_const(0),
    symbol_at(int_type2(), "c:test.c@F@main@w", symbol_renaming_level::level1)};
  REQUIRE_FALSE(e.state().constant_propagation(
    constant_array2tc(int_array(2), mutable_elems)));
}

TEST_CASE(
  "an aggregate-typed leaf is decided by constant_propagation, not by "
  "immutability",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc base = symbol_at(
    pair_struct(), "c:test.c@F@main@VAR", symbol_renaming_level::level2);

  // The scalar guard: an L2 symbol of array type is still refused, so
  // array_may_propagate keeps deciding the infinite-size modelling arrays.
  const expr2tc inf_array = symbol_at(
    array_type2tc(int_type2(), expr2tc(), true),
    "c:test.c@F@main@__ESBMC_alloc",
    symbol_renaming_level::level2);
  REQUIRE_FALSE(e.state().constant_propagation(inf_array));

  // But a fixed-index read out of one carries only the scalar leaf. This is
  // the shape the peel permits rather than one observed in a C program: such
  // reads normally carry a symbolic index, which the peel rejects.
  REQUIRE(e.state().constant_propagation(
    with_field(base, "r", index_of(inf_array, int_const(3)))));
}

TEST_CASE(
  "a chain of symbolic updates is carried only up to the bound",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc arr = symbol_at(
    int_array(4096), "c:test.c@F@main@A", symbol_renaming_level::level2);

  // Carrying a chain is quadratic in its length, so the updates only
  // is_immutable_value accepts are counted and capped (#7597).
  expr2tc chain = arr;
  for (unsigned i = 0; i < 128; i++)
    chain = with_index(chain, int_const(i), nondet_int_symbol());
  REQUIRE(e.state().constant_propagation(chain));

  REQUIRE_FALSE(e.state().constant_propagation(
    with_index(chain, int_const(128), nondet_int_symbol())));

  // Updates that propagate on their own are not counted, so a chain of them
  // is carried at any length -- pre-#7597 behaviour is unchanged.
  expr2tc literals = arr;
  for (unsigned i = 0; i < 400; i++)
    literals = with_index(literals, int_const(i), int_const(i));
  REQUIRE(e.state().constant_propagation(literals));
}
