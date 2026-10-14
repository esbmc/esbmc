#include <assert.h>
#include <stdint.h>
int main()
{
  uint8_t a, b;
  __CPROVER_assume(a == 200 && b == 100);
  uint16_t wide = (uint16_t)((uint16_t)a * (uint16_t)b);
  uint8_t hi = (uint8_t)(wide >> 8);
  assert(hi == 77);   // WRONG: high byte of 20000 is 78
  return 0;
}
