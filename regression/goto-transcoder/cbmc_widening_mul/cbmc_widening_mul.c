#include <assert.h>
#include <stdint.h>
int main()
{
  uint8_t a, b;
  __CPROVER_assume(a == 200 && b == 100);   // 20000 = 0x4E20
  uint16_t wide = (uint16_t)((uint16_t)a * (uint16_t)b);
  uint8_t lo = (uint8_t)wide;                // 0x20 = 32
  uint8_t hi = (uint8_t)(wide >> 8);         // 0x4E = 78
  assert(lo == 32 && hi == 78);
  assert(((uint16_t)hi << 8 | lo) == wide);
  return 0;
}
