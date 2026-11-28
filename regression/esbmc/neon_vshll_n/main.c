// vshll_n and vshll_high_n widen each lane and shift it left, by up to the
// narrow element width. The expected values are what the same code gives
// natively on AArch64.
#include <arm_neon.h>
#include <assert.h>

int main(void)
{
  uint8x8_t u8 = {0, 1, 200, 255, 4, 5, 6, 7};
  uint16x8_t a = vshll_n_u8(u8, 3);
  assert(a[1] == 8 && a[2] == 1600 && a[3] == 2040);
  uint16x8_t b = vshll_n_u8(u8, 8);
  assert(b[1] == 256 && b[3] == 65280);

  int8x8_t s8 = {-128, -1, 100, 127, 0, 5, -6, 7};
  int16x8_t c = vshll_n_s8(s8, 8);
  assert(c[0] == -32768 && c[1] == -256 && c[3] == 32512);

  int16x8_t s16 = {0, -1, 30000, -32768, 32767, 5, -6, 7};
  int32x4_t d = vshll_n_s16(vget_low_s16(s16), 16);
  assert(d[1] == -65536 && d[3] == -2147483647 - 1);
  int32x4_t e = vshll_high_n_s16(s16, 5);
  assert(e[0] == 1048544 && e[2] == -192 && e[3] == 224);

  uint16x8_t u16 = {1, 2, 3, 4, 65535, 6, 7, 8};
  uint32x4_t f = vshll_high_n_u16(u16, 16);
  assert(f[0] == 4294901760u && f[1] == 393216);

  int32x2_t s32 = {-2147483647 - 1, 3};
  int64x2_t g = vshll_n_s32(s32, 32);
  assert(g[0] == (-9223372036854775807ll - 1) && g[1] == 12884901888ll);
  uint8x16_t q8 = vcombine_u8(u8, vdup_n_u8(255));
  uint16x8_t h = vshll_high_n_u8(q8, 1);
  assert(h[0] == 510 && h[7] == 510);
  return 0;
}
