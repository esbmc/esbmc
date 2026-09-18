#include <assert.h>
#include <stdint.h>

unsigned long nondet_ulong(void);
char a[64], b[64];

/* ptr_int_mul_unknown_alias with &b[0] registered before p is reconstructed. */
int main(void)
{
  __ESBMC_assume((uintptr_t)&b[0] != 0);
  unsigned long u = nondet_ulong();
  char *p = (char *)(u * 8 - (uintptr_t)&a[0]);
  if ((uintptr_t)p == (uintptr_t)&b[0])
  {
    *p = 3;
    assert(b[0] == 0);
  }
  return 0;
}
