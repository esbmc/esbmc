// #7907: a vector written and read back through a pointer to a
// nondeterministic element of an array of vectors.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

int nondet_int(void);

int main(void)
{
  v4i a[3];
  int k = nondet_int();
  __ESBMC_assume(k >= 0 && k < 3);
  v4i *q = a + k;
  *q = (v4i){1, 2, 3, 4};
  v4i c = *q;
  assert(c[3] == 4);
  return 0;
}
