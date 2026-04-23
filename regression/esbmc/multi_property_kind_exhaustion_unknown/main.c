#include <assert.h>

extern int nondet_int(void);
extern _Bool nondet_bool(void);

/* Regression for the --multi-property + --k-induction exhaustion
   behaviour:

   Before the fix, once BC recorded a per-claim violation, the
   k-loop would run to --max-k-step with FC/IS skipped on every
   subsequent round (the "!is_bcv" guard), then fall through to the
   hard-coded "Unable to prove or falsify the program" /
   "VERIFICATION UNKNOWN" exit.  The user was left with an
   UNKNOWN verdict AND no indication that a definitive bug had
   actually been found at some earlier k.

   After the fix, the same program still exhausts the k budget
   (IS genuinely cannot discharge the walk-bounded claim below),
   but the final log distinguishes "exhausted with earlier
   per-claim violations recorded" from a plain "nothing found".
*/
int main(void)
{
  int x = nondet_int();
  int sum = 0;

  /* Claim 1 -- BC falsifies at k=1. */
  assert(x != 42);

  /* Bounded random walk: after k iterations, |sum| <= k. */
  while (nondet_bool())
  {
    if (nondet_bool())
      sum++;
    else
      sum--;
  }

  /* Claim 2 -- provable only with a walk-bound invariant ESBMC's IS
     will not find.  BC at k <= 3 cannot reach sum == 100 either,
     so this claim stays active across all k iterations and forces
     the outer loop to fall through with "violations recorded but
     remaining claims unresolved". */
  assert(sum != 100);

  return 0;
}
