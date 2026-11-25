// #10887: a NaN bit pattern moved through a float must come back unchanged.
#include <assert.h>
#include <stdint.h>
#include <string.h>

unsigned nondet_uint(void);
unsigned long long nondet_ulonglong(void);

int main(void)
{
  uint32_t x = nondet_uint();
  float f;
  memcpy(&f, &x, sizeof f);
  uint32_t y;
  memcpy(&y, &f, sizeof y);
  assert(x == y);

  uint64_t a = nondet_ulonglong();
  double d;
  memcpy(&d, &a, sizeof d);
  uint64_t b;
  memcpy(&b, &d, sizeof b);
  assert(a == b);
  return 0;
}
