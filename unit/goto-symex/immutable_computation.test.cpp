/*******************************************************************
 Module: goto_symex_statet::constant_propagation on immutable computations

 A `with`-chain update value or an aggregate-literal element is carried when
 is_immutable_computation accepts it: an immutable leaf (is_immutable_value),
 or a pure bitvector computation over such leaves -- the cell-packing idiom
 `(short)((x >> 16) & 0xFFFF)`. These pin the individual arms an end-to-end
 test cannot separate: which opcodes count, the is_bv_type gate, the
 constant-leaf exclusions, and that a pointer leaf is refused.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

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

/** An assigned-once (L2) leaf of type @p t. */
expr2tc l2(const type2tc &t, const char *name = "c:test.c@F@main@v")
{
  return symbol_at(t, name, symbol_renaming_level::level2);
}

expr2tc int_const(int v)
{
  return constant_int2tc(int_type2(), BigInt(v));
}

/** A struct with one field named `f` of type @p field, so a computation of
 *  that type can be tested as the update value of a `with` chain. */
type2tc box(const type2tc &field)
{
  std::vector<type2tc> members{field};
  std::vector<irep_idt> names{"f"};
  return struct_type2tc(members, names, names, "box");
}

/** `with(base_of_type_box(field), "f", value)`: propagates exactly when
 *  is_immutable_computation carries `value`, since the base is an
 *  unpropagatable L2 symbol. */
expr2tc carried(const type2tc &field, const expr2tc &value)
{
  const expr2tc base =
    symbol_at(box(field), "c:test.c@F@main@B", symbol_renaming_level::level2);
  const type2tc str_type = array_type2tc(get_uint8_type(), gen_ulong(2), false);
  return with2tc(
    base->type,
    base,
    constant_string2tc(str_type, "f", constant_string_kindt::DEFAULT),
    value);
}
} // namespace

TEST_CASE(
  "each accepted bitvector opcode over immutable leaves is carried",
  "[symex][immutable-computation]")
{
  engine e;
  const expr2tc a = l2(int_type2(), "c:test.c@F@main@a");
  const expr2tc b = l2(int_type2(), "c:test.c@F@main@b");
  const expr2tc sh = int_const(3);

  const expr2tc ops[] = {
    bitand2tc(int_type2(), a, b),
    bitor2tc(int_type2(), a, b),
    bitxor2tc(int_type2(), a, b),
    shl2tc(int_type2(), a, sh),
    lshr2tc(int_type2(), a, sh),
    ashr2tc(int_type2(), a, sh),
    add2tc(int_type2(), a, b),
    sub2tc(int_type2(), a, b),
  };

  for (const expr2tc &op : ops)
  {
    CAPTURE(get_expr_id(op));
    REQUIRE(e.state().constant_propagation(carried(int_type2(), op)));
  }
}

TEST_CASE(
  "the cell-packing idiom (s)((x >> 16) & 0xFFFF) is carried",
  "[symex][immutable-computation]")
{
  engine e;
  const expr2tc x = l2(int_type2(), "c:test.c@F@main@x");
  const expr2tc packed = typecast2tc(
    get_int16_type(),
    bitand2tc(
      int_type2(), lshr2tc(int_type2(), x, int_const(16)), int_const(0xFFFF)));
  REQUIRE(e.state().constant_propagation(carried(get_int16_type(), packed)));
}

TEST_CASE(
  "an opcode outside the whitelist is not carried",
  "[symex][immutable-computation]")
{
  engine e;
  const expr2tc a = l2(int_type2(), "c:test.c@F@main@a");
  const expr2tc b = l2(int_type2(), "c:test.c@F@main@b");

  // mul/div/modulus are not pure cell-packing bit-ops: not carried as an
  // immutable computation, and not constant here either (the leaves are
  // symbols, so constant_propagation's arithmetic path declines too).
  for (const expr2tc &op :
       {mul2tc(int_type2(), a, b),
        div2tc(int_type2(), a, b),
        modulus2tc(int_type2(), a, b)})
  {
    CAPTURE(get_expr_id(op));
    REQUIRE_FALSE(e.state().constant_propagation(carried(int_type2(), op)));
  }
}

TEST_CASE(
  "the is_bv_type gate rejects a non-bitvector computation",
  "[symex][immutable-computation]")
{
  engine e;
  const type2tc fbv = fixedbv_type2tc(32, 16);
  const expr2tc fa = l2(fbv, "c:test.c@F@main@fa");
  const expr2tc fb = l2(fbv, "c:test.c@F@main@fb");

  // A bare fixedbv leaf is immutable (a scalar symbol), so it is carried.
  REQUIRE(e.state().constant_propagation(carried(fbv, fa)));

  // But a fixedbv *computation* is not a bitvector one: the gate refuses it,
  // even though its leaves are immutable.
  REQUIRE_FALSE(
    e.state().constant_propagation(carried(fbv, add2tc(fbv, fa, fb))));
}

TEST_CASE(
  "a constant scalar leaf is carried but a constant aggregate is not",
  "[symex][immutable-computation]")
{
  engine e;

  // A constant scalar routes through as immutable.
  REQUIRE(e.state().constant_propagation(carried(int_type2(), int_const(7))));

  // A constant aggregate literal must go through constant_propagation instead,
  // so array_may_propagate keeps its say -- is_immutable_computation excludes
  // it.
  const type2tc arr2 =
    array_type2tc(int_type2(), constant_int2tc(size_type2(), BigInt(2)), false);
  std::vector<expr2tc> elems{int_const(0), int_const(1)};
  const expr2tc const_arr = constant_array2tc(arr2, elems);
  // Carried as an aggregate field only via constant_propagation's array path,
  // never as an "immutable computation": a symbolic sibling element would keep
  // it out (covered by the array-literal cases), which is the point of the
  // exclusion.
  REQUIRE(e.state().constant_propagation(const_arr));
}

TEST_CASE("a pointer leaf is left symbolic", "[symex][immutable-computation]")
{
  engine e;
  const type2tc ptr = pointer_type2tc(int_type2());

  // A pointer's SSA symbol never changes, but carrying the pointer value lets
  // a dereference or iterator arithmetic through it fold to a target a later
  // aliased or symbolic-index store cannot invalidate. So an assigned-once
  // pointer leaf is refused, on the leaf and inside a carried aggregate.
  const expr2tc p = l2(ptr, "c:test.c@F@main@p");
  REQUIRE_FALSE(e.state().constant_propagation(carried(ptr, p)));

  std::vector<expr2tc> fields{p};
  REQUIRE_FALSE(
    e.state().constant_propagation(constant_struct2tc(box(ptr), fields)));

  // A constant-propagatable pointer (NULL) still carries: the exclusion is of
  // the *symbol* leaf, not of every pointer.
  const expr2tc null_ptr =
    symbol2tc(ptr, "NULL", symbol_renaming_level::level0, 0, 0, 0, 0);
  REQUIRE(e.state().constant_propagation(carried(ptr, null_ptr)));
}
