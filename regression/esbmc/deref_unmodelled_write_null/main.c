#include <stdint.h>

_Bool nondet_bool(void);
unsigned long nondet_ulong(void);
char a[64];

/* A value set holding NULL beside unknown: NULL keeps its own claim, and the
 * unresolved branch is reported separately. */
int main(void)
{
  unsigned long u = nondet_ulong();
  char *p = nondet_bool() ? (char *)0 : (char *)(u * 8 - (uintptr_t)&a[0]);
  *p = 3;
  return 0;
}
