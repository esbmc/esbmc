/*******************************************************************
 Module: L2 renaming on the real renaming::level2t

 Tier B of docs/roadmap/goto-symex-verification-plan.md (M1, and H-B4 in M4).

 The subject is the shipped class: the `renaming::level2t` owned by a real
 `execution_statet`, its real `current_names` (a
 `std::unordered_map<name_record, valuet, name_rec_hash>`), and the real
 `make_assignment` -> `rename` -> `coveredinbees` chain. Only the input symbols
 are constructed here.

 Discharges:
   I1  per key, make_assignment publishes count_before + 1 and stores it.
   I2  the key `coveredinbees` recomputes is the caller's key, so the
       `valuet &entry` make_assignment holds addresses the entry that is
       updated (finding R3).
   I3  rename is idempotent: an already-L2 symbol comes back unchanged.
   I4  get_original_name inverts rename, and a definition's renaming level
       never drops below L2 (H-B4). Listed unenforced in the plan's §4.2.
   R3  the *memory-safety* half of the finding, tested rather than assumed:
       [unord.req.general]/9 — "Rehashing invalidates iterators [...] but does
       not invalidate pointers or references to elements" — so an insert inside
       the nested lookup cannot dangle the held reference. See §15.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>

#include <goto-symex/reachability_tree.h>
#include <goto-symex/renaming.h>
#include <irep2/irep2_expr.h>
#include <util/lang/c_types.h>
#include <util/symtab/namespace.h>

#include "ssa_validator.h"
#include "../testing-utils/goto_factory.h"

namespace
{
/** Owns everything a real execution state needs to stay alive. */
class engine
{
public:
  explicit engine(std::string src = "int main(void) { int x = 0; return x; }")
    : source(std::move(src)),
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
    opts.set_option("unwind", "4");
    rt.setup_for_new_explore();
  }

  renaming::level2t &level2()
  {
    return rt.get_cur_state().get_active_state().level2;
  }

  std::shared_ptr<symex_target_equationt> run()
  {
    auto eq = std::dynamic_pointer_cast<symex_target_equationt>(
      rt.get_next_formula().target);
    REQUIRE(eq != nullptr);
    symex_ssa::require_well_formed(*eq);
    return eq;
  }

private:
  std::string source;
  program prog;
  namespacet ns;
  optionst opts;
  reachability_treet rt;
};

expr2tc l1_symbol(
  const std::string &name,
  unsigned l1_num = 0,
  unsigned thread_num = 0,
  symbol_renaming_level lev = symbol_renaming_level::level1)
{
  return symbol2tc(int_type2(), irep_idt(name), lev, l1_num, 0, thread_num, 0);
}

/** make_assignment mutates its lhs in place; return the index it published. */
unsigned publish(renaming::level2t &l2, const expr2tc &l1_sym)
{
  expr2tc lhs = l1_sym;
  l2.make_assignment(lhs, expr2tc(), expr2tc());
  return to_symbol2t(lhs).level2_num;
}

renaming::level2t::name_record key_of(const expr2tc &l1_sym)
{
  return renaming::level2t::name_record(to_symbol2t(l1_sym));
}

/** rename() mutates in place; return the renamed copy. */
expr2tc renamed(renaming::level2t &l2, const expr2tc &e)
{
  expr2tc out = e;
  l2.rename(out);
  return out;
}

/** Strip an L2 symbol to L0 the way the engine does: level2t drops to L1,
 *  level1t drops that to L0. */
expr2tc original_of(const expr2tc &e)
{
  expr2tc out = e;
  renaming::renaming_levelt::get_original_name(
    out, symbol_renaming_level::level1);
  renaming::renaming_levelt::get_original_name(
    out, symbol_renaming_level::level0);
  return out;
}

bool same_symbol_identity(const expr2tc &a, const expr2tc &b)
{
  const symbol2t &x = to_symbol2t(a);
  const symbol2t &y = to_symbol2t(b);
  return x.thename == y.thename && x.rlevel == y.rlevel &&
         x.level1_num == y.level1_num && x.level2_num == y.level2_num &&
         x.thread_num == y.thread_num && x.node_num == y.node_num &&
         x.type == y.type;
}
} // namespace

TEST_CASE(
  "make_assignment publishes a fresh increasing L2 index",
  "[symex][renaming]")
{
  engine e;
  renaming::level2t &l2 = e.level2();
  const expr2tc sym = l1_symbol("c:test.c@F@main@x");

  for (unsigned expected = 1; expected <= 5; expected++)
  {
    const unsigned published = publish(l2, sym);
    REQUIRE(published == expected);
    // I2: the entry coveredinbees updated is the one keyed by the caller's key.
    REQUIRE(l2.current_names.at(key_of(sym)).count == published);
  }
}

TEST_CASE("a fresh key costs exactly one map entry", "[symex][renaming]")
{
  engine e;
  renaming::level2t &l2 = e.level2();
  const expr2tc sym = l1_symbol("c:test.c@F@main@y");

  const size_t before = l2.current_names.size();
  REQUIRE(publish(l2, sym) == 1);

  // If the key recomputed inside coveredinbees differed from the caller's, the
  // nested current_names[...] would default-insert a second entry (I2 / R3).
  REQUIRE(l2.current_names.size() == before + 1);

  REQUIRE(publish(l2, sym) == 2);
  REQUIRE(l2.current_names.size() == before + 1);
}

TEST_CASE("keys differing in one field do not alias", "[symex][renaming]")
{
  engine e;
  renaming::level2t &l2 = e.level2();

  const expr2tc base = l1_symbol("c:test.c@F@main@z", 0, 0);
  const expr2tc other_l1 = l1_symbol("c:test.c@F@main@z", 1, 0);
  const expr2tc other_thread = l1_symbol("c:test.c@F@main@z", 0, 1);
  const expr2tc other_level =
    l1_symbol("c:test.c@F@main@z", 0, 0, symbol_renaming_level::level1_global);

  for (const expr2tc &sym : {base, other_l1, other_thread, other_level})
    REQUIRE(publish(l2, sym) == 1);

  REQUIRE(publish(l2, base) == 2);
  REQUIRE(l2.current_names.at(key_of(other_l1)).count == 1);
  REQUIRE(l2.current_names.at(key_of(other_thread)).count == 1);
  REQUIRE(l2.current_names.at(key_of(other_level)).count == 1);
}

TEST_CASE(
  "a reference into current_names survives the rehash R3 fears",
  "[symex][renaming]")
{
  engine e;
  renaming::level2t &l2 = e.level2();
  const expr2tc sym = l1_symbol("c:test.c@F@main@held");

  REQUIRE(publish(l2, sym) == 1);
  const renaming::level2t::valuet *entry = &l2.current_names.at(key_of(sym));
  const size_t buckets_before = l2.current_names.bucket_count();

  // Force at least one rehash — this is the event R3 names as the trigger for
  // a dangling `valuet &entry` in make_assignment.
  for (unsigned i = 0; i < 256; i++)
    publish(l2, l1_symbol("c:test.c@F@main@filler" + std::to_string(i)));
  REQUIRE(l2.current_names.bucket_count() > buckets_before);

  // [unord.req.general]/9: rehashing does not invalidate pointers or
  // references to elements. The held reference still addresses the live entry.
  REQUIRE(&l2.current_names.at(key_of(sym)) == entry);
  REQUIRE(entry->count == 1);
  REQUIRE(publish(l2, sym) == 2);
  REQUIRE(entry->count == 2);
}

// ---------------------------------------------------------------------------
// H-B4: renaming round-trip (I3, I4)
// ---------------------------------------------------------------------------

TEST_CASE("rename is idempotent on an L2 symbol (I3)", "[symex][renaming]")
{
  engine e;
  renaming::level2t &l2 = e.level2();
  expr2tc sym = l1_symbol("c:test.c@F@main@idem");

  // Without an entry, rename's early return fires because the map is empty
  // and the idempotence claim is vacuous.
  l2.make_assignment(sym, expr2tc(), expr2tc());
  REQUIRE(to_symbol2t(sym).rlevel == symbol_renaming_level::level2);

  const expr2tc once = renamed(l2, sym);
  const expr2tc twice = renamed(l2, once);
  CHECK(same_symbol_identity(once, sym));
  CHECK(same_symbol_identity(twice, once));
}

TEST_CASE(
  "rename of an L1 symbol reaches a fixed point (I3)",
  "[symex][renaming]")
{
  engine e;
  renaming::level2t &l2 = e.level2();
  const expr2tc l1 = l1_symbol("c:test.c@F@main@fix");

  expr2tc published = l1;
  l2.make_assignment(published, expr2tc(), expr2tc());

  // A second application moving the index would let a read observe a value
  // other than the one the first rename selected.
  const expr2tc once = renamed(l2, l1);
  REQUIRE(to_symbol2t(once).rlevel == symbol_renaming_level::level2);
  REQUIRE(to_symbol2t(once).level2_num == to_symbol2t(published).level2_num);
  CHECK(same_symbol_identity(renamed(l2, once), once));
}

TEST_CASE("get_original_name inverts rename (I4)", "[symex][renaming]")
{
  engine e;
  renaming::level2t &l2 = e.level2();

  for (const symbol_renaming_level lev :
       {symbol_renaming_level::level1, symbol_renaming_level::level1_global})
  {
    const expr2tc l1 = l1_symbol("c:test.c@F@main@trip", 3, 2, lev);
    const expr2tc l2_sym = renamed(l2, l1);

    // The round trip has to land back on whichever level it started from.
    expr2tc back = l2_sym;
    renaming::renaming_levelt::get_original_name(
      back, symbol_renaming_level::level1);
    CHECK(same_symbol_identity(back, l1));
  }
}

TEST_CASE("stripping to L0 keeps name and type (I4)", "[symex][renaming]")
{
  engine e;
  renaming::level2t &l2 = e.level2();
  const expr2tc l1 = l1_symbol("c:test.c@F@main@strip", 7, 1);
  expr2tc sym = l1;
  l2.make_assignment(sym, expr2tc(), expr2tc());

  const expr2tc l0 = original_of(sym);
  const symbol2t &stripped = to_symbol2t(l0);

  // Non-vacuity: same_symbol_identity would pass on anything if L0 == L2.
  REQUIRE_FALSE(same_symbol_identity(l0, sym));

  // The type surviving is the load-bearing half: rewriting it would make the
  // L0 form name a different object.
  CHECK(stripped.rlevel == symbol_renaming_level::level0);
  CHECK(stripped.thename == to_symbol2t(l1).thename);
  CHECK(stripped.type == l1->type);
  CHECK(stripped.level1_num == 0);
  CHECK(stripped.level2_num == 0);
  CHECK(stripped.thread_num == 0);
  CHECK(stripped.node_num == 0);

  // Idempotent: already-L0 is a fixed point.
  CHECK(same_symbol_identity(original_of(l0), l0));
}

TEST_CASE(
  "every equation definition strips cleanly to L0 (I4)",
  "[symex][renaming]")
{
  // Repeated and nested calls give callee locals distinct L1 activations;
  // without them the level1_num half is only ever checked against zero.
  engine e(R"(
int nondet_int(void);
int global;
int twice(int a) { int t = a + a; return t; }
int sum_to(int n) { int acc = n <= 0 ? 0 : n + sum_to(n - 1); return acc; }
int main(void)
{
  int x = nondet_int();
  global = x;
  if (nondet_int() > 0)
    x = twice(x);
  x = twice(x) + twice(x + 1);
  x += sum_to(3);
  for (int i = 0; i < 3; i++)
    x += i;
  return x + global;
}
)");

  auto eq = e.run();
  unsigned checked = 0;
  unsigned with_activation = 0;

  for (const auto &step : eq->SSA_steps)
  {
    if (!step.is_assignment() || !is_symbol2t(step.lhs))
      continue;

    // I4: a definition below L2 could be renamed again by a later read, which
    // would pick a different index than the one this step defined.
    const symbol2t &lhs = to_symbol2t(step.lhs);
    REQUIRE(
      (lhs.rlevel == symbol_renaming_level::level2 ||
       lhs.rlevel == symbol_renaming_level::level2_global));

    const expr2tc l0 = original_of(step.lhs);
    const symbol2t &stripped = to_symbol2t(l0);
    CHECK(stripped.rlevel == symbol_renaming_level::level0);
    CHECK(stripped.thename == lhs.thename);
    CHECK(stripped.type == step.lhs->type);
    CHECK(stripped.level1_num == 0);
    CHECK(stripped.level2_num == 0);
    CHECK(stripped.thread_num == 0);
    CHECK(stripped.node_num == 0);
    CHECK(same_symbol_identity(original_of(l0), l0));
    checked++;
    if (lhs.level1_num != 0)
      with_activation++;
  }

  // Guard against an empty sweep, and against one of only zero-L1 definitions.
  REQUIRE(checked > 0);
  REQUIRE(with_activation > 0);
}

TEST_CASE(
  "a default-constructed L2 name_record is fully initialised (R10)",
  "[symex][renaming]")
{
  using record = renaming::level2t::name_record;

  // R10: the four fields and the derived hash were left indeterminate by
  // `= default`. No in-tree site default-constructs one today, so the check
  // that carries information is that the defaults are *consistent*: `compare`
  // short-circuits on `hash`, so a hash that does not match the fields would
  // make two equal records compare unequal.
  record a;
  record b;
  REQUIRE(a == b);
  REQUIRE(a.compare(b) == 0);

  REQUIRE(a.base_name == irep_idt());
  REQUIRE(a.lev == symbol_renaming_level::level0);
  REQUIRE(a.l1_num == 0);
  REQUIRE(a.t_num == 0);

  // The invariant the hash exists to serve: equal fields imply equal hash, so
  // a record built from an L0 symbol matching the defaults must land on the
  // same fast-path key rather than merely comparing equal field-by-field.
  expr2tc sym = l1_symbol("", 0, 0, symbol_renaming_level::level0);
  record from_symbol(to_symbol2t(sym));
  REQUIRE(from_symbol.hash == a.hash);
  REQUIRE(from_symbol == a);
}
