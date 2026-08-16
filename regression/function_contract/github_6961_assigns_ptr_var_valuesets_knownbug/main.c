/* The value-set skip is tested on the widened place, which rescues a decayed
 * array argument (github_6961_assigns_ptr_param_valuesets_fail). A pointer
 * variable stays a pointer through the widening, so it is skipped and nothing
 * is havocked at all: the ensures then contradicts b[0]'s pre-call value, the
 * path dies, and the assertion below is never reached.
 *
 * --add-symex-value-sets is default-enabled by --k-induction, --inductive-step
 * and --loop-invariant, so this shape is silently unchecked in all of those. */
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
