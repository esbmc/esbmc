#include <stdint.h>
#include <assert.h>

unsigned long nondet_ulong(void);

char a[64];
char c = 0;

int main(void)
{
  // Same guard as ptr_int_mul_unknown_alias, for a pointer that names no
  // object at all: its numeric value is below 64, so it is wild (#6545).
  unsigned long u = nondet_ulong();
  char *p = (char *)(u * 8 - (uintptr_t)&a[0]);
  if ((uintptr_t)p < 64)
  {
    *p = 3;
    assert(c == 0);
  }
  return 0;
}
