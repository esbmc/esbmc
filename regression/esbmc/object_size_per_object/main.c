#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

/* A pointer that may address either of two objects has the size of the one it
 * addresses, not of whichever its value set lists first. */
_Bool nondet_bool(void);
unsigned nondet_uint(void);
size_t __ESBMC_builtin_object_size(const void *, int);

int main(void)
{
  char small[1];
  char big[100];
  char *p = nondet_bool() ? small : big + 50;
  if (p == big + 50)
  {
    assert(__ESBMC_builtin_object_size(p, 0) == 100);
    assert(__ESBMC_builtin_object_size(p, 3) == 50);
  }
  else
  {
    assert(__ESBMC_builtin_object_size(p, 0) == 1);
    assert(__ESBMC_builtin_object_size(p, 3) == 1);
  }

  unsigned n = nondet_uint();
  __ESBMC_assume(n > 0 && n < 50);
  char *heap = malloc(n);
  char *q = nondet_bool() ? heap : big;
  if (heap && q == heap)
    assert(__ESBMC_builtin_object_size(q, 0) == n);
  free(heap);
  return 0;
}
