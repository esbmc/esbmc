#include <assert.h>

extern int nondet_int(void);
extern _Bool nondet_bool(void);

/* Minimal regression for the --multi-property + --incremental-bmc
   verdict loss bug: the outer k-loop used to skip the forward
   condition whenever BC had already found a violation ("is_bcv &&
   FC" guard) which, combined with the hard-coded "VERIFICATION
   UNKNOWN" fall-through, silently downgraded a real bug detection
   to UNKNOWN.

   After the fix: multi_property_check clears the violated claim at
   k=1, the k-loop sees zero active assertions and short-circuits via
   conclude() -> VERIFICATION FAILED.
*/
int main(void)
{
  int x = nondet_int();

  /* BC at k=1 finds this when x == 42. */
  assert(x != 42);

  /* Nondet-controlled unbounded loop: without early termination the
     pre-fix code would otherwise run to max_k_step and fall through
     to the hard-coded UNKNOWN branch. */
  while (nondet_bool())
    x++;

  return 0;
}
