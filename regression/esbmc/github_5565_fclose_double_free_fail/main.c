/* #5565 boundary: skipping free() for non-heap streams must not weaken
 * double-free detection.  An fopen'd stream is heap memory, so closing it
 * twice must still be caught -- the __ESBMC_is_dynamic guard suppresses the
 * free only for streams ESBMC never allocated, not for fopen'd ones. */
#include <stdio.h>

int main(void)
{
  FILE *f = fopen("x", "r");
  if (f)
  {
    fclose(f);
    fclose(f); /* double close of a heap stream -> double free */
  }
  return 0;
}
