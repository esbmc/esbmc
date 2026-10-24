#include <assert.h>
#include <stddef.h>

/* A pointer that may address either of two objects has the size of the one it
 * addresses, not of whichever its value set lists first. */
_Bool nondet_bool(void);

int main(void)
{
  char small[1];
  char big[100];
  char *p = nondet_bool() ? small : big + 50;
  size_t n = __CPROVER_OBJECT_SIZE(p);
  if (p == big + 50)
    assert(n == 100);
  else
    assert(n == 1);

  int x;
  void *q = nondet_bool() ? (void *)big : (void *)&x;
  if (q == big)
    assert(__CPROVER_OBJECT_SIZE(q) == 100);
  return 0;
}
