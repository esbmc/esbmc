// __builtin_shufflevector of a bit-cast vector and of a vector with nondet
// lanes, neither of which symex sees as a constant vector.
#include <assert.h>

typedef unsigned char v8u8 __attribute__((vector_size(8)));
typedef unsigned short v4u16 __attribute__((vector_size(8)));

unsigned char nondet_uchar(void);

int main(void)
{
  v4u16 lo = {0x0102, 0x0304, 0x0506, 0x0708};
  v4u16 hi = {0x0a0b, 0x0a0b, 0x0a0b, 0x0a0b};
  v8u8 r = __builtin_shufflevector((v8u8)lo, (v8u8)hi, 0, 2, 4, 6, 8, 10, 12, 14);
  assert(r[0] == 0x02 && r[3] == 0x08 && r[4] == 0x0b);

  v8u8 a, b = {0};
  for (int i = 0; i < 8; i++)
    a[i] = nondet_uchar();
  v8u8 z = __builtin_shufflevector(a, b, 7, 8, 0, -1, 1, 2, 3, 4);
  assert(z[0] == a[7] && z[1] == 0 && z[2] == a[0] && z[7] == a[4]);
  return 0;
}
