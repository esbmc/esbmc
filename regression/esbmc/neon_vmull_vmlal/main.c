// vmull, vmull_high, vmlal, vmlal_high and vmlsl for each widening, and the
// carry-less vmull_p8. The expected values are what the same code gives
// natively on AArch64.
#include <arm_neon.h>
#include <assert.h>

int main(void)
{
  uint8x8_t au8 = {0, 1, 200, 255, 16, 5, 6, 7};
  uint8x8_t bu8 = {9, 0, 201, 255, 16, 5, 60, 1};
  uint16x8_t mu8 = vmull_u8(au8, bu8);
  assert(mu8[0] == 0 && mu8[2] == 40200 && mu8[3] == 65025 && mu8[6] == 360);

  int8x8_t as8 = {-128, -1, 100, 127, 0, 5, -6, 7};
  int8x8_t bs8 = {-128, 2, -100, 127, 3, 5, 6, -7};
  int16x8_t ms8 = vmull_s8(as8, bs8);
  assert(ms8[0] == 16384 && ms8[1] == -2 && ms8[2] == -10000 && ms8[7] == -49);

  uint16x4_t au16 = {65535, 2, 300, 4}, bu16 = {65535, 3, 300, 5};
  uint32x4_t mu16 = vmull_u16(au16, bu16);
  assert(mu16[0] == 4294836225u && mu16[2] == 90000);
  int16x4_t as16 = {-32768, 1, -300, 4}, bs16 = {-32768, -1, 300, 5};
  int32x4_t ms16 = vmull_s16(as16, bs16);
  assert(ms16[0] == 1073741824 && ms16[1] == -1 && ms16[2] == -90000);

  uint32x2_t au32 = {4000000000u, 7}, bu32 = {4000000000u, 6};
  uint64x2_t mu32 = vmull_u32(au32, bu32);
  assert(mu32[0] == 16000000000000000000ull && mu32[1] == 42);
  int32x2_t as32 = {-2000000000, 3}, bs32 = {2000000000, -3};
  int64x2_t ms32 = vmull_s32(as32, bs32);
  assert(ms32[0] == -4000000000000000000ll && ms32[1] == -9);

  poly8x8_t ap8 = {3, 0xff, 0x80, 7, 0, 1, 2, 5};
  poly8x8_t bp8 = {3, 0xff, 0x80, 7, 9, 1, 2, 5};
  poly16x8_t mp8 = vmull_p8(ap8, bp8);
  assert(mp8[0] == 5 && mp8[1] == 0x5555 && mp8[2] == 0x4000 && mp8[3] == 21);
  assert(mp8[4] == 0 && mp8[7] == 17);

  uint8x16_t qa = vcombine_u8(bu8, au8), qb = vcombine_u8(au8, bu8);
  uint16x8_t hi = vmull_high_u8(qa, qb);
  assert(hi[2] == 40200 && hi[3] == 65025);

  uint16x8_t acc = vdupq_n_u16(1000);
  uint16x8_t ml = vmlal_u8(acc, au8, bu8);
  assert(ml[0] == 1000 && ml[2] == 41200 && ml[3] == 489);
  uint16x8_t mlh = vmlal_high_u8(acc, qa, qb);
  assert(mlh[2] == 41200);
  int32x4_t mls = vmlsl_s16(vdupq_n_s32(5), as16, bs16);
  assert(mls[0] == -1073741819 && mls[1] == 6 && mls[2] == 90005);
  return 0;
}
