/* #2797: soundness guard for the fseeko operational model. fseeko must be a
 * nondet over-approximation that can fail (-1), not an unsound "always succeeds"
 * stub. If it were modelled as `return 0`, this assertion would wrongly hold and
 * ESBMC would miss bugs in callers' seek-error-handling paths. The nondet model
 * means r can be nonzero, so the assertion fails as expected. */
#include <stdio.h>

int main(void)
{
  FILE *f = fopen("file", "w");
  if (!f)
    return 0;
  int r = fseeko(f, 0, SEEK_SET);
  assert(r == 0); /* nondet return may be nonzero: fseeko can fail */
  fclose(f);
  return 0;
}
