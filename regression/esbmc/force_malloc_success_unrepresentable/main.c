#include <stdlib.h>
#include <assert.h>

size_t nondet_size(void);

int main()
{
  // --force-malloc-success cannot fail the allocation, so the unrepresentable
  // size is excluded by assumption instead and this assert stays unreachable.
  // Residual of R25; see docs/roadmap/goto-symex-verification-plan.md.
  size_t n = nondet_size();
  __ESBMC_assume(n >= 0xFFFFFFFFFFFFFFF0UL);
  char *b = malloc(n);
  if (b)
    b[0] = 1;
  assert(0);
}
