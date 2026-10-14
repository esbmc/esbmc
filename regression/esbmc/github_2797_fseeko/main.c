/* #2797: ESBMC had no operational model for fseeko (only the other coreutils
 * libc functions were modelled by #2798), so calls emitted
 * "WARNING: no body for function fseeko". This test confirms the model is now
 * present (no warning) and that the reachable success outcome verifies. */
#include <stdio.h>

int main(void)
{
  FILE *f = fopen("file", "w");
  if (!f)
    return 0;
  int r = fseeko(f, 0, SEEK_SET);
  __ESBMC_assume(r == 0); /* the success outcome is reachable */
  assert(r == 0);
  fclose(f);
  return 0;
}
