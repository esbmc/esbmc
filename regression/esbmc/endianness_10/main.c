#include <stdint.h>
#include <assert.h>

typedef union
{
  uint16_t a[4];
  uint64_t u64;
} U;

int main()
{
  U u;
  u.u64 = 0x1122334455667788ULL;
  /* little-endian: a[0] spans bytes 0-1 (byte 0 is LSB) */
  assert(u.a[0] == 0x7788);
  assert(u.a[1] == 0x5566);
  assert(u.a[2] == 0x3344);
  assert(u.a[3] == 0x1122);
  return 0;
}
