// #7907: a vector read and written through a pointer at a nondeterministic,
// suitably aligned offset into an int array.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

int nondet_int(void);

int main(void)
{
  int buf[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int j = nondet_int();
  __ESBMC_assume(j == 0 || j == 4);
  v4i c = *(v4i *)(buf + j);
  assert(c[3] == j + 3);
  *(v4i *)(buf + j) = (v4i){9, 9, 9, 9};
  assert(buf[j + 1] == 9);
  return 0;
}
