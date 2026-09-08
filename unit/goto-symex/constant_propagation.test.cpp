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

#include <string>
#include <vector>

#include <goto-symex/reachability_tree.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/config/config.h>
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

/** `struct { int i; int *p; }`, a counter beside a pointer member. */
type2tc counter_and_pointer_struct()
{
  std::vector<type2tc> members{int_type2(), pointer_type2tc(int_type2())};
  std::vector<irep_idt> names{"i", "p"};
  return struct_type2tc(members, names, names, "held");
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
expr2tc with_field(
  const expr2tc &source,
  const std::string &field,
  const expr2tc &value)
{
  const type2tc str_type =
    array_type2tc(get_uint8_type(), gen_ulong(field.size() + 1), false);
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

/** Restores a global option however the case leaves: REQUIRE throws, and a
 *  leaked k-induction would fail every later case in this binary instead. */
class scoped_option
{
public:
  scoped_option(const char *name, bool value)
    : name(name), saved(config.options.get_bool_option(name))
  {
    config.options.set_option(name, value);
  }
  ~scoped_option()
  {
    config.options.set_option(name, saved);
  }

private:
  const char *name;
  bool saved;
};

/** `struct { int i; int a[4]; }`, a counter beside an array member. */
type2tc counter_and_array_struct()
{
  std::vector<type2tc> members{int_type2(), int_array(4)};
  std::vector<irep_idt> names{"i", "a"};
  return struct_type2tc(members, names, names, "held_array");
}

/** `struct { int i; float _Complex z; }`: a member a read *is* offered for --
 *  the object is a struct -- but whose complex type the acceptance re-test then
 *  refuses, so the rebuild is dropped after an element was pinned. */
type2tc counter_and_complex_struct()
{
  std::vector<type2tc> members{
    int_type2(),
    complex_type2tc(float_type2()),
  };
  std::vector<irep_idt> names{"i", "z"};
  return struct_type2tc(members, names, names, "held_complex");
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

TEST_CASE("a pointer leaf is not immutable", "[symex][constant-propagation]")
{
  engine e;
  const expr2tc base = symbol_at(
    counter_and_pointer_struct(),
    "c:test.c@F@main@VAR",
    symbol_renaming_level::level2);

  // The pointer is assigned-once, so it is as fixed as any other leaf, and it
  // is still refused: carrying one resolves a later dereference against the
  // wrong object (#7605).
  const expr2tc ptr = symbol_at(
    pointer_type2tc(int_type2()),
    "c:test.c@F@main@q",
    symbol_renaming_level::level2);
  REQUIRE_FALSE(e.state().constant_propagation(with_field(base, "p", ptr)));

  // Its integer sibling in the same struct still carries, so what the chain
  // refuses is the type and not the write.
  REQUIRE(e.state().constant_propagation(with_field(base, "i", l2_int())));
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
  // goto_symex_state.cpp's symbolic_chain_bound, which is file-local there.
  constexpr unsigned bound = 1024;

  engine e;
  const expr2tc arr = symbol_at(
    int_array(bound + 2), "c:test.c@F@main@A", symbol_renaming_level::level2);

  // A carried chain is walked again at every write, so the updates only
  // is_immutable_value accepts are counted and capped (#7597).
  expr2tc chain = arr;
  for (unsigned i = 0; i < bound; i++)
    chain = with_index(chain, int_const(i), nondet_int_symbol());
  REQUIRE(e.state().constant_propagation(chain));

  REQUIRE_FALSE(e.state().constant_propagation(
    with_index(chain, int_const(bound), nondet_int_symbol())));

  // Updates that propagate on their own are not counted, so a chain of them
  // is carried at any length -- pre-#7597 behaviour is unchanged.
  expr2tc literals = arr;
  for (unsigned i = 0; i < bound + 2; i++)
    literals = with_index(literals, int_const(i), int_const(i));
  REQUIRE(e.state().constant_propagation(literals));
}

/* The rest cover pin_symbolic_updates. #7605 widened which *values* may be
 * carried; an operator around an immutable read is still not one of them, so
 * the object -- and the sibling counter in it -- was dropped as before. Rather
 * than widen the class again, a refused write is re-offered as a read of the
 * name the assignment has just defined, which denotes it exactly. */

TEST_CASE(
  "an operator around an immutable read is not itself immutable",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc io = symbol_at(
    pair_struct(), "c:test.c@F@main@IO", symbol_renaming_level::level2);
  const expr2tc var = symbol_at(
    pair_struct(), "c:test.c@F@main@VAR", symbol_renaming_level::level2);

  // The bare read #7605 admits.
  REQUIRE(
    e.state().constant_propagation(with_field(var, "r", member_of(io, "i"))));

  // Wrapped in any operator it is refused, and the whole object goes with it.
  const expr2tc sum = add2tc(int_type2(), member_of(io, "i"), int_const(1));
  REQUIRE_FALSE(e.state().constant_propagation(with_field(var, "r", sum)));
}

TEST_CASE(
  "a refused chain update is pinned to a read of the assigned name",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc io = symbol_at(
    pair_struct(), "c:test.c@F@main@IO", symbol_renaming_level::level2);
  const expr2tc var = symbol_at(
    pair_struct(), "c:test.c@F@main@VAR", symbol_renaming_level::level2);
  const expr2tc sum = add2tc(int_type2(), member_of(io, "i"), int_const(1));

  // `VAR.i = 3; VAR.r = IO.i + 1;` as symex lowers it.
  const expr2tc rhs = with_field(with_field(var, "i", int_const(3)), "r", sum);
  const expr2tc pinned = e.state().pin_symbolic_updates(rhs, var);

  REQUIRE_FALSE(is_nil_expr(pinned));
  REQUIRE(e.state().constant_propagation(pinned));

  // The refused value now reads out of VAR itself, ...
  REQUIRE(to_with2t(pinned).update_value == member_of(var, "r"));
  // ... and the sibling counter folds again, which is the point.
  REQUIRE(member_of(pinned, "i")->simplify() == int_const(3));
}

TEST_CASE(
  "a refused literal element is pinned too",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc io = symbol_at(
    pair_struct(), "c:test.c@F@main@IO", symbol_renaming_level::level2);
  const expr2tc var = symbol_at(
    pair_struct(), "c:test.c@F@main@VAR", symbol_renaming_level::level2);
  const expr2tc sum = add2tc(int_type2(), member_of(io, "i"), int_const(1));

  // do_simplify folds a `with` over a propagated literal back into a literal,
  // so the same write reaches assignment() in this shape as well.
  const expr2tc literal =
    constant_struct2tc(pair_struct(), std::vector<expr2tc>{int_const(3), sum});
  const expr2tc pinned = e.state().pin_symbolic_updates(literal, var);

  REQUIRE_FALSE(is_nil_expr(pinned));
  REQUIRE(e.state().constant_propagation(pinned));
  REQUIRE(
    to_constant_struct2t(pinned).datatype_members[1] == member_of(var, "r"));
  REQUIRE(member_of(pinned, "i")->simplify() == int_const(3));
}

TEST_CASE(
  "an array element is pinned to an index read",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc arr =
    symbol_at(int_array(2), "c:test.c@F@main@A", symbol_renaming_level::level2);
  const expr2tc sum = add2tc(int_type2(), nondet_int_symbol(), int_const(1));

  const expr2tc literal =
    constant_array2tc(int_array(2), std::vector<expr2tc>{int_const(3), sum});
  const expr2tc pinned = e.state().pin_symbolic_updates(literal, arr);

  REQUIRE_FALSE(is_nil_expr(pinned));
  REQUIRE(e.state().constant_propagation(pinned));
  REQUIRE(
    to_constant_array2t(pinned).datatype_members[1] ==
    index_of(arr, gen_ulong(1)));
}

TEST_CASE(
  "an element write at a symbolic index is not pinned",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc io = symbol_at(
    pair_struct(), "c:test.c@F@main@IO", symbol_renaming_level::level2);
  const expr2tc refused = add2tc(int_type2(), member_of(io, "i"), int_const(1));
  const expr2tc arr =
    symbol_at(int_array(2), "c:test.c@F@main@A", symbol_renaming_level::level2);

  // A write at a constant index is re-offered as a read of that element, ...
  REQUIRE_FALSE(is_nil_expr(e.state().pin_symbolic_updates(
    with_index(arr, int_const(1), refused), arr)));

  // ... but only a constant index reads back to an immutable value, so a
  // symbolic one ends the chain and the object stays unpropagated as before.
  // Two guards hold this and either alone suffices: read_of_field declines to
  // offer the read, and is_immutable_value refuses it at the acceptance
  // re-test. The case flips only with both removed.
  REQUIRE(is_nil_expr(e.state().pin_symbolic_updates(
    with_index(arr, nondet_int_symbol(), refused), arr)));
}

TEST_CASE(
  "a write into an array member is pinned to a read of the member",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc var = symbol_at(
    counter_and_array_struct(),
    "c:test.c@F@main@VAR",
    symbol_renaming_level::level2);

  // `VAR.a[VAR.i] = x + 1` lowers to a `with` on VAR whose update value is
  // itself a `with` over the array member -- aggregate-typed, so neither
  // constant_propagation nor a scalar-only immutability test accepts it, and
  // the counter beside it used to be dropped with the object
  // (InduByte/esbmc-evaluation#4).
  const expr2tc member_read = member_of(var, "a", int_array(4));
  const expr2tc refused = with_index(
    member_read,
    nondet_int_symbol(),
    add2tc(int_type2(), nondet_int_symbol(), int_const(1)));

  const expr2tc pinned = e.state().pin_symbolic_updates(
    with_field(with_field(var, "i", int_const(3)), "a", refused), var);

  REQUIRE_FALSE(is_nil_expr(pinned));
  REQUIRE(e.state().constant_propagation(pinned));
  REQUIRE(to_with2t(pinned).update_value == member_read);
  // ... and the sibling counter folds again, which is the point.
  REQUIRE(member_of(pinned, "i")->simplify() == int_const(3));
}

TEST_CASE(
  "a pointer member read is still not carried",
  "[symex][constant-propagation]")
{
  // The aggregate widening above admits fixed-size struct and array reads only.
  // A pointer stays out: carrying one resolves a later dereference against the
  // wrong object, the false alarm in
  // regression/esbmc-cpp/cpp/github_5868_list_iterator_adl.
  engine e;
  const expr2tc io = symbol_at(
    counter_and_pointer_struct(),
    "c:test.c@F@main@IO",
    symbol_renaming_level::level2);
  const expr2tc var = symbol_at(
    counter_and_pointer_struct(),
    "c:test.c@F@main@VAR",
    symbol_renaming_level::level2);

  const expr2tc ptr_read = member_of(io, "p", pointer_type2tc(int_type2()));
  REQUIRE_FALSE(e.state().constant_propagation(
    with_field(with_field(var, "i", int_const(3)), "p", ptr_read)));
}

TEST_CASE(
  "a merged object is pinned member by member",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc var = symbol_at(
    pair_struct(), "c:test.c@F@main@VAR", symbol_renaming_level::level2);
  const expr2tc cond = symbol_at(
    get_bool_type(), "nondet$symex::g", symbol_renaming_level::level0);

  // phi_function's shape after `if (g) VAR.r = ...;` inside a loop: the counter
  // agrees on both arms, the written member does not. constant_propagation
  // carries an `if` at no arm, so the whole object used to be dropped
  // (InduByte/esbmc-evaluation#5).
  const expr2tc taken = constant_struct2tc(
    pair_struct(), std::vector<expr2tc>{int_const(3), int_const(7)});
  const expr2tc other = constant_struct2tc(
    pair_struct(), std::vector<expr2tc>{int_const(3), int_const(9)});
  const expr2tc phi = if2tc(pair_struct(), cond, taken, other);

  REQUIRE_FALSE(e.state().constant_propagation(phi));

  const expr2tc pinned = e.state().pin_symbolic_updates(phi, var);
  REQUIRE_FALSE(is_nil_expr(pinned));
  REQUIRE(e.state().constant_propagation(pinned));

  // The member the branch disagrees on reads out of VAR itself, ...
  REQUIRE(
    to_constant_struct2t(pinned).datatype_members[1] == member_of(var, "r"));
  // ... and the counter both arms agree on folds again, which is the point.
  REQUIRE(member_of(pinned, "i")->simplify() == int_const(3));
}

TEST_CASE(
  "a merged array is pinned element by element",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc arr =
    symbol_at(int_array(2), "c:test.c@F@main@A", symbol_renaming_level::level2);
  const expr2tc cond = symbol_at(
    get_bool_type(), "nondet$symex::g", symbol_renaming_level::level0);

  // The same merge over a bare array: element 0 is the counter both arms agree
  // on, element 1 is the one the branch wrote.
  const expr2tc taken = constant_array2tc(
    int_array(2), std::vector<expr2tc>{int_const(3), int_const(7)});
  const expr2tc other = constant_array2tc(
    int_array(2), std::vector<expr2tc>{int_const(3), int_const(9)});
  const expr2tc phi = if2tc(int_array(2), cond, taken, other);

  REQUIRE_FALSE(e.state().constant_propagation(phi));

  const expr2tc pinned = e.state().pin_symbolic_updates(phi, arr);
  REQUIRE_FALSE(is_nil_expr(pinned));
  REQUIRE(e.state().constant_propagation(pinned));
  REQUIRE(
    to_constant_array2t(pinned).datatype_members[1] ==
    index_of(arr, gen_ulong(1)));
  REQUIRE(index_of(pinned, gen_ulong(0))->simplify() == int_const(3));
}

TEST_CASE(
  "a merge both arms agree on needs no pinning",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc var = symbol_at(
    pair_struct(), "c:test.c@F@main@VAR", symbol_renaming_level::level2);
  const expr2tc cond = symbol_at(
    get_bool_type(), "nondet$symex::g", symbol_renaming_level::level0);

  // Nothing to re-offer: every member is the branch's value either way, so the
  // rebuild would carry no read and pinning declines.
  const expr2tc same = constant_struct2tc(
    pair_struct(), std::vector<expr2tc>{int_const(3), int_const(7)});

  REQUIRE(is_nil_expr(e.state().pin_symbolic_updates(
    if2tc(pair_struct(), cond, same, same), var)));
}

TEST_CASE(
  "pinning declines when it would change nothing",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc var = symbol_at(
    pair_struct(), "c:test.c@F@main@VAR", symbol_renaming_level::level2);

  // Every update already propagates, so the value assignment() recorded is
  // already the right one and there is nothing to re-offer.
  REQUIRE(is_nil_expr(
    e.state().pin_symbolic_updates(with_field(var, "i", int_const(3)), var)));

  // A shape that is neither a chain nor an aggregate literal is left alone.
  REQUIRE(is_nil_expr(e.state().pin_symbolic_updates(int_const(3), var)));
}

TEST_CASE(
  "pinning is off under the incremental strategies",
  "[symex][constant-propagation]")
{
  engine e;
  const expr2tc io = symbol_at(
    pair_struct(), "c:test.c@F@main@IO", symbol_renaming_level::level2);
  const expr2tc var = symbol_at(
    pair_struct(), "c:test.c@F@main@VAR", symbol_renaming_level::level2);

  // A literal, not a chain: the `with` branch of constant_propagation opts out
  // under these strategies on its own, so the chain path would decline at its
  // own acceptance test. The literal branch has no such opt-out to inherit,
  // and this is the shape that needs the explicit one.
  const expr2tc literal = constant_struct2tc(
    pair_struct(),
    std::vector<expr2tc>{
      int_const(3), add2tc(int_type2(), member_of(io, "i"), int_const(1))});

  REQUIRE_FALSE(is_nil_expr(e.state().pin_symbolic_updates(literal, var)));

  {
    const scoped_option k("k-induction", true);
    REQUIRE(is_nil_expr(e.state().pin_symbolic_updates(literal, var)));
  }

  REQUIRE_FALSE(is_nil_expr(e.state().pin_symbolic_updates(literal, var)));
}

TEST_CASE(
  "a pinned read constant_propagation still refuses ends the chain",
  "[symex][constant-propagation]")
{
  // The acceptance re-test is what contains the feature: pinning offers a read,
  // it does not decide that the read may be carried. A `_Complex` member is
  // offered as `member(VAR, "z")`, which is neither a scalar nor a fixed-size
  // aggregate update, so the rebuild is refused and the object stays
  // unpropagated -- exactly as before #7597.
  engine e;
  const expr2tc io = symbol_at(
    pair_struct(), "c:test.c@F@main@IO", symbol_renaming_level::level2);
  const expr2tc var = symbol_at(
    counter_and_complex_struct(),
    "c:test.c@F@main@VAR",
    symbol_renaming_level::level2);

  const expr2tc refused = typecast2tc(
    complex_type2tc(float_type2()),
    add2tc(int_type2(), member_of(io, "i"), int_const(1)));
  REQUIRE(is_nil_expr(e.state().pin_symbolic_updates(
    with_field(with_field(var, "i", int_const(3)), "z", refused), var)));
}

TEST_CASE("a large array is left unpropagated", "[symex][constant-propagation]")
{
  // goto_symex_state.cpp's pinned_array_bound, which is file-local there.
  constexpr unsigned bound = 64;

  engine e;
  const expr2tc io = symbol_at(
    pair_struct(), "c:test.c@F@main@IO", symbol_renaming_level::level2);
  const expr2tc refused = add2tc(int_type2(), member_of(io, "i"), int_const(1));

  // Carrying a big array costs a rebuild per element write and buys nothing:
  // nothing reads a sibling of an array a loop is filling.
  const expr2tc small = symbol_at(
    int_array(bound), "c:test.c@F@main@A", symbol_renaming_level::level2);
  REQUIRE_FALSE(is_nil_expr(e.state().pin_symbolic_updates(
    with_index(small, int_const(0), refused), small)));

  const expr2tc big = symbol_at(
    int_array(bound + 1), "c:test.c@F@main@B", symbol_renaming_level::level2);
  REQUIRE(is_nil_expr(e.state().pin_symbolic_updates(
    with_index(big, int_const(0), refused), big)));
}
