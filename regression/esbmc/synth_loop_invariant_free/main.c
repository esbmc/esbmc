/* An OTHER instruction in the body is not inert: `free(p)` lowers to
 * `OTHER FREE(...)`, and the schema's havoc names variables, so it cannot
 * express what the free did to the heap. Summarising the body around it cut
 * the loop and proved this use-after-free safe. */
#include <stdlib.h>

unsigned int nondet_uint(void);

int main(void)
{
  unsigned int n = nondet_uint();
  __ESBMC_assume(n >= 1 && n <= 4);

  int *p = malloc(sizeof(int));
  if (p == 0)
    return 0;

  unsigned int i = 0;
  unsigned int s = 0;

  while (i < n)
  {
    free(p);
    s = s + 2;
    i = i + 1;
  }

  *p = 3;
  return 0;
}
