// #7907: a vector read and written through a pointer to it.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

int nondet_int(void);

int main(void)
{
  v4i s = {1, 2, 3, 4};
  v4i *p = &s;
  v4i c = *p;
  assert(c[0] == 1);

  *p = (v4i){5, 6, 7, 8};
  assert(s[3] == 8);

  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 4);
  int x = (*p)[i];
  assert(x == i + 5);
  return 0;
}
