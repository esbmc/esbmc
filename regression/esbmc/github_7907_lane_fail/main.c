// #7907: the lane written through the pointer is the one it names.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

int nondet_int(void);

int main(void)
{
  v4i s = {1, 2, 3, 4};
  v4i *p = &s;
  int k = nondet_int();
  __ESBMC_assume(k >= 0 && k < 4);
  (*p)[k] = 9;
  assert(s[1] == 2);
  return 0;
}
