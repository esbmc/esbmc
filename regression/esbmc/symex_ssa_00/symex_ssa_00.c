/*
 * Tier-A harness template — docs/roadmap/goto-symex-verification-plan.md, M0.
 *
 * SYMEX-HARNESS-TARGET: src/goto-symex/renaming.cpp::renaming::level2t::make_assignment
 * SYMEX-HARNESS-SHA256: 2c4c9016c75c0ef2e613cb9c9c0519cd212a3d5644821f7bc49a977bf9f54d37
 * SYMEX-HARNESS-TARGET: src/goto-symex/renaming.cpp::renaming::level2t::coveredinbees
 * SYMEX-HARNESS-SHA256: 8c0c0e7b6e3a84b7e8c864b80188e178b3446bb1860db841aa0f477d153303d9
 *
 * Discharges (partially, at the smoke level): I1 / P4 / P8 / P12 — the L2
 * counter published by make_assignment is fresh, strictly increasing per key,
 * and never wraps.
 *
 * Assumptions (§6.1 rule 2 — each cites the symbol that establishes it):
 *   key < MAP_CAP
 *     src/goto-symex/renaming.h::renaming::level2t::name_record — the L2 key is
 *     a value type over (base_name, rlevel, l1_num, t_num); the harness
 *     enumerates a bounded key space, per §6.1 rule 4 (bound the shape).
 *   before < UINT_MAX
 *     src/goto-symex/renaming.cpp::renaming::level2t::coveredinbees — its
 *     SYMEX_INVARIANT(entry.count <= count) is the engine-side statement that
 *     counts only ever grow, checked in release builds since M3. Discharged by
 *     Tier B (H-B1) rather than assumed there.
 *
 * Variants — one source, selected by -D on test.desc line 3, so the twins
 * cannot drift from the kernel they perturb:
 *   (none)                  regression/esbmc/symex_ssa_00        SUCCESSFUL
 *   SYMEX_HARNESS_PERTURB   regression/esbmc/symex_ssa_00_fail   FAILED (§6.1 r5)
 *   SYMEX_HARNESS_PROBE     regression/esbmc/symex_ssa_00_probe  FAILED (§6.1 r6)
 */

#include <limits.h>

#define MAP_CAP 4
#define SEQ_LEN 4

unsigned nondet_uint(void);

/* renaming::level2t::valuet, reduced to the one field the property observes
 * (§6.2 stub inventory: node ids, the propagated constant and the expression
 * payload are not read by I1, and the latter is irep2's obligation). */
struct valuet
{
  unsigned count;
};

static struct valuet current_names[MAP_CAP];

/* renaming::level2t::make_assignment — the counter algebra only. */
static unsigned make_assignment(unsigned key)
{
  struct valuet *entry = &current_names[key];
#ifdef SYMEX_HARNESS_PERTURB
  unsigned published = entry->count;
#else
  unsigned published = entry->count + 1;
#endif
  entry->count = published;
  return published;
}

int main(void)
{
  unsigned keys[SEQ_LEN];
  unsigned published[SEQ_LEN];

  for (unsigned i = 0; i < MAP_CAP; i++)
    current_names[i].count = 0;

  for (unsigned i = 0; i < SEQ_LEN; i++)
  {
    unsigned key = nondet_uint();
    __ESBMC_assume(key < MAP_CAP);

    unsigned before = current_names[key].count;
    __ESBMC_assume(before < UINT_MAX);

    keys[i] = key;
    published[i] = make_assignment(key);

    __ESBMC_assert(
      published[i] == before + 1, "I1: L2 index advances by exactly one");
    __ESBMC_assert(
      current_names[key].count == published[i],
      "I1: the map holds the index just published");

    for (unsigned j = 0; j < i; j++)
      __ESBMC_assert(
        keys[j] != key || published[j] != published[i],
        "I1: a published (key, L2 index) pair is never republished");
  }

#ifdef SYMEX_HARNESS_PROBE
  __ESBMC_assert(0, "reachability probe");
#endif

  return 0;
}
