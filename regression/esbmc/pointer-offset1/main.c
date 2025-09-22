#include <assert.h>
#include <stdlib.h>

int main()
{
  int *p, *q;
  int x = 10, y = -1;

  // Different stack objects → offset 0 for both
  p = &x;
  q = &y;
  assert(__ESBMC_POINTER_OFFSET(p) == __ESBMC_POINTER_OFFSET(q));

  int a[10], b[5];
  
  // Array bases also start at offset 0
  assert(__ESBMC_POINTER_OFFSET(a) == __ESBMC_POINTER_OFFSET(b));
  
  // Array indexing → offsets increase correctly
  assert(__ESBMC_POINTER_OFFSET(&a[0]) == 0);
  assert(__ESBMC_POINTER_OFFSET(&a[1]) == sizeof(int));
  assert(__ESBMC_POINTER_OFFSET(&a[9]) == 9 * sizeof(int));
  
  // Pointer arithmetic → consistency
  p = a;
  assert(__ESBMC_POINTER_OFFSET(p + 2) == 2 * sizeof(int));
  assert(__ESBMC_POINTER_OFFSET(p - 1) == -1 * (int)sizeof(int));
  
  // Struct members → different offsets
  struct S { char c; int d; } s;
  assert(__ESBMC_POINTER_OFFSET(&s.c) == 0);

  // Null pointer → defined as offset 0 in ESBMC
  int *nullp = 0;
  assert(__ESBMC_POINTER_OFFSET(nullp) == 0);

  return 0;
}

