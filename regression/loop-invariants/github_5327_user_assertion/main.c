/* Companion to function_contract/github_5327: scoping the vacuity probe to
 * user-facing claims must not stop it firing on a user assertion. The two
 * assumes contradict, so the assert below is discharged on a dead path. */
#include <assert.h>

extern int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x > 0);
  __ESBMC_assume(x < 0);

  int i = 0;
  __ESBMC_loop_invariant(i <= 3);
  while (i < 3)
  {
    i = i + 1;
  }

  assert(x == 42);
  return 0;
}
