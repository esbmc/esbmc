#include <stdlib.h>
#include <assert.h>

size_t nondet_size(void);

int main()
{
  // --force-malloc-success removes the ordinary may-fail outcome, but an
  // unrepresentable size still yields NULL rather than being assumed away, so
  // execution continues and this assert is reachable. Was R25's residual; see
  // docs/roadmap/goto-symex-verification-plan.md.
  size_t n = nondet_size();
  __ESBMC_assume(n >= 0xFFFFFFFFFFFFFFF0UL);
  char *b = malloc(n);
  if (b)
    b[0] = 1;
  assert(0);
}
