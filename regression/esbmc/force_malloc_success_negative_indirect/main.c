/* github_1631_nondet_compact's contract, laundered through a size_t variable:
   under --force-malloc-success a negative-size request must leave the code
   after it reachable. The size symex_mem sees is then the plain symbol `n`
   rather than a typecast, so bounding it at PTRDIFF_MAX and exempting the
   negative case by inspecting the argument's syntax misses this spelling and
   proves the assert vacuously. See
   docs/roadmap/goto-symex-verification-plan.md, R38's residual. */
#include <assert.h>
#include <stdlib.h>

int main(void)
{
  int a = nondet_int();
  __ESBMC_assume(a < 0);

  size_t n = a;
  void *b = malloc(n);

  assert(0);
}
