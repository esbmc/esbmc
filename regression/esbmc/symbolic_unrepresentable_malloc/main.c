#include <stdlib.h>
#include <assert.h>

size_t nondet_size(void);

int main()
{
  // No flag is needed: the pointer is used, so the slicer keeps the
  // allocation and the default configuration goes vacuous unless the
  // unrepresentable request is failed rather than laid out.
  size_t n = nondet_size();
  __ESBMC_assume(n >= 0xFFFFFFFFFFFFFFF0UL);
  char *b = malloc(n);
  if (b)
    b[0] = 1;
  assert(0);
}
