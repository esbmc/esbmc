// vbsl takes each bit from the second operand where the mask is set and from
// the third where it is clear. The expected values are what the same code
// gives natively on AArch64.
#include <arm_neon.h>
#include <assert.h>

int main(void)
{
  uint8x16_t m = vdupq_n_u8(0xf0);
  uint8x16_t a = vdupq_n_u8(0xab), b = vdupq_n_u8(0xcd);
  uint8x16_t r = vbslq_u8(m, a, b);
  assert(r[0] == 0xad && r[15] == 0xad);

  uint32x2_t m32 = {0xffffffffu, 0};
  float32x2_t x = {1.5f, 2.5f}, y = {-1.0f, -2.0f};
  float32x2_t f = vbsl_f32(m32, x, y);
  assert(f[0] == 1.5f && f[1] == -2.0f);

  uint16x8_t m16 = vcgtq_u16(vdupq_n_u16(5), (uint16x8_t){1, 9, 2, 8, 3, 7, 4, 6});
  int16x8_t s = vbslq_s16(m16, vdupq_n_s16(-1), vdupq_n_s16(1));
  assert(s[0] == -1 && s[1] == 1 && s[6] == -1 && s[7] == 1);
  return 0;
}
