/*******************************************************************
 Module: L2 renaming on the real renaming::level2t

 Tier B of docs/roadmap/goto-symex-verification-plan.md (milestone M1).

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

#include "../testing-utils/goto_factory.h"

namespace
{
/** Owns everything a real execution state needs to stay alive. */
class engine
{
public:
  engine()
    : prog(goto_factory::get_goto_functions(
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

  renaming::level2t &level2()
  {
    return rt.get_cur_state().get_active_state().level2;
  }

private:
  std::string source = "int main(void) { int x = 0; return x; }";
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
