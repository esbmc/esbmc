/* The value-set skip is about assigning a pointer, not about writing through
 * one, so it is tested on the place actually written -- after widening and
 * after following a pointer parameter. Tested earlier it swallowed the whole
 * frame: nothing was havocked, the ensures contradicted b[0]'s pre-call value
 * and the assertion below was never reached.
 *
 * --add-symex-value-sets is default-enabled by --k-induction, --inductive-step
 * and --loop-invariant, so this shape was silently unchecked in all of those. */
#define N 4

void clr(int *p)
{
  __ESBMC_assigns(p);
  __ESBMC_ensures(p[0] == 0);

  for (int i = 0; i < N; i++)
    p[i] = 0;
}

int main(void)
{
  int b[N];
  int *q = b;
  b[0] = 1;
  clr(q);
  __ESBMC_assert(0, "the path must stay alive to reach this");
  return 0;
}
