#include <stdlib.h>
#include <assert.h>

size_t nondet_size(void);

int main()
{
  // R25's residual, still open. Under --force-malloc-success a symbolic size is
  // bounded by assumption rather than by a NULL branch -- branching costs
  // 21s -> >400s on github_1352-success-32bit -- so a request the address space
  // cannot lay out is excluded instead of failed, and this assert is unreachable.
  // See docs/roadmap/goto-symex-verification-plan.md, R25 and R38.
  size_t n = nondet_size();
  __ESBMC_assume(n >= 0xFFFFFFFFFFFFFFF0UL);
  char *b = malloc(n);
  if (b)
    b[0] = 1;
  assert(0);
}
