// With nondet lanes, each vmull_u8 lane is the 16-bit product and each
// vmlal_s16 lane the accumulated 32-bit product.
#include <arm_neon.h>
#include <assert.h>

unsigned char nondet_uchar(void);
short nondet_short(void);
int nondet_int(void);

int main(void)
{
  uint8x8_t a, b;
  for (int i = 0; i < 8; i++)
  {
    a[i] = nondet_uchar();
    b[i] = nondet_uchar();
  }
  uint16x8_t m = vmull_u8(a, b);
  assert(m[3] == (unsigned short)(a[3] * b[3]));
  assert(m[7] == (unsigned short)(a[7] * b[7]));

  int16x4_t x = {nondet_short(), 0, 0, nondet_short()};
  int16x4_t y = {nondet_short(), 0, 0, nondet_short()};
  int32x4_t acc = {nondet_int(), 0, 0, 0};
  __ESBMC_assume(acc[0] > -1000000 && acc[0] < 1000000);
  int32x4_t r = vmlal_s16(acc, x, y);
  assert(r[0] == acc[0] + x[0] * y[0] && r[3] == x[3] * y[3]);
  return 0;
}
