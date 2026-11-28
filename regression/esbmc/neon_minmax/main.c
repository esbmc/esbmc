// vmax/vmin, D and Q, for the integer and float element types. A float lane
// follows FMAX/FMIN: NaN wins, and max(-0, +0) is +0. The expected values are
// what the same code gives natively on AArch64.
#include <arm_neon.h>
#include <assert.h>
#include <math.h>

int main(void)
{
  uint8x16_t au8 = {0, 1, 200, 3, 250, 5, 6, 7, 128, 9, 10, 255, 12, 13, 14, 15};
  uint8x16_t bu8 = {9, 0, 201, 2, 7, 5, 60, 1, 127, 200, 3, 254, 15, 14, 13, 12};
  uint8x16_t mx = vmaxq_u8(au8, bu8), mn = vminq_u8(au8, bu8);
  assert(mx[0] == 9 && mx[2] == 201 && mx[4] == 250 && mx[8] == 128);
  assert(mx[11] == 255 && mx[15] == 15);
  assert(mn[0] == 0 && mn[2] == 200 && mn[4] == 7 && mn[8] == 127);
  assert(mn[11] == 254 && mn[15] == 12);

  int8x8_t as8 = {0, -1, 100, -128, 127, 5, -6, 7};
  int8x8_t bs8 = {1, -2, -100, 127, -128, 5, 6, -7};
  int8x8_t s8x = vmax_s8(as8, bs8), s8n = vmin_s8(as8, bs8);
  assert(s8x[1] == -1 && s8x[2] == 100 && s8x[3] == 127 && s8x[6] == 6);
  assert(s8n[1] == -2 && s8n[2] == -100 && s8n[3] == -128 && s8n[7] == -7);

  uint16x4_t au16 = {0, 60000, 3, 65535}, bu16 = {1, 59999, 4, 65534};
  uint16x4_t u16x = vmax_u16(au16, bu16);
  assert(u16x[0] == 1 && u16x[1] == 60000 && u16x[3] == 65535);
  int16x8_t as16 = {0, -1, 30000, -32768, 32767, 5, -6, 7};
  int16x8_t bs16 = {1, -2, -30000, 32767, -32768, 5, 6, -7};
  int16x8_t s16n = vminq_s16(as16, bs16);
  assert(s16n[2] == -30000 && s16n[3] == -32768 && s16n[4] == -32768);

  uint32x4_t au32 = {0, 4000000000u, 3, 7}, bu32 = {1, 3999999999u, 4, 6};
  uint32x4_t u32n = vminq_u32(au32, bu32);
  assert(u32n[0] == 0 && u32n[1] == 3999999999u && u32n[3] == 6);
  int32x2_t as32 = {-2000000000, 2000000000}, bs32 = {2000000000, -7};
  int32x2_t s32x = vmax_s32(as32, bs32);
  assert(s32x[0] == 2000000000 && s32x[1] == 2000000000);

  float32x4_t af = {1.5f, -0.0f, 0.0f, NAN}, bf = {-2.0f, 0.0f, -0.0f, 1.0f};
  float32x4_t fx = vmaxq_f32(af, bf), fn = vminq_f32(af, bf);
  assert(fx[0] == 1.5f && fn[0] == -2.0f);
  assert(fx[1] == 0 && !signbit(fx[1]) && fn[1] == 0 && signbit(fn[1]));
  assert(!signbit(fx[2]) && signbit(fn[2]));
  assert(isnan(fx[3]) && isnan(fn[3]));
  float64x2_t ad = {1.0, -3.0}, bd = {2.0, -4.0};
  float64x2_t dx = vmaxq_f64(ad, bd);
  assert(dx[0] == 2.0 && dx[1] == -3.0);
  float32x2_t ah = {1.0f, 2.0f}, bh = {3.0f, -1.0f};
  float32x2_t hn = vmin_f32(ah, bh);
  assert(hn[0] == 1.0f && hn[1] == -1.0f);
  return 0;
}
