#include <assert.h>
#include <stdint.h>

_Bool nondet_bool(void);
unsigned long nondet_ulong(void);
char a[64], b[64];

/* The value set holds b and unknown; a write that lands on b is modelled. */
int main(void)
{
  unsigned long u = nondet_ulong();
  char *p = nondet_bool() ? &b[0] : (char *)(u * 8 - (uintptr_t)&a[0]);
  if (p == &b[0])
  {
    *p = 3;
    assert(b[0] == 3);
  }
  return 0;
}
