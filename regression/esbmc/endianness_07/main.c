#include <stdint.h>
#include <assert.h>

typedef union
{
  unsigned char a[8];
  unsigned long long u64;
} U;

U init()
{
  U r;
  r.u64 = 0x1122334455667788ULL;
  return r;
}

int main()
{
  U u = init();
  /* big-endian: byte 0 (lowest address) is MSB */
  assert(u.a[0] == 0x11);
  assert(u.a[1] == 0x22);
  assert(u.a[2] == 0x33);
  assert(u.a[3] == 0x44);
  assert(u.a[4] == 0x55);
  assert(u.a[5] == 0x66);
  assert(u.a[6] == 0x77);
  assert(u.a[7] == 0x88);
  return 0;
}
