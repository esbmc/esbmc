// vget_lane/vset_lane for every element type, D and Q. The expected values
// are what the same code gives natively on AArch64.
#include <arm_neon.h>
#include <assert.h>

int main(void)
{
  uint8x16_t qu8 = {0, 1, 2, 200, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255};
  assert(vgetq_lane_u8(qu8, 3) == 200);
  assert(vgetq_lane_u8(qu8, 15) == 255);
  qu8 = vsetq_lane_u8(77, qu8, 15);
  assert(qu8[15] == 77 && qu8[14] == 14);

  int8x8_t ds8 = {-1, 2, -128, 4, 5, 6, 7, 127};
  assert(vget_lane_s8(ds8, 2) == -128);
  ds8 = vset_lane_s8(-7, ds8, 7);
  assert(ds8[7] == -7 && ds8[6] == 7);

  uint16x8_t qu16 = {0, 1, 2, 3, 4, 5, 60000, 7};
  assert(vgetq_lane_u16(qu16, 6) == 60000);
  int16x4_t ds16 = {-32768, 1, 2, 3};
  assert(vget_lane_s16(ds16, 0) == -32768);
  ds16 = vset_lane_s16(-5, ds16, 3);
  assert(ds16[3] == -5 && ds16[0] == -32768);

  uint32x4_t qu32 = {0, 4000000000u, 2, 3};
  assert(vgetq_lane_u32(qu32, 1) == 4000000000u);
  qu32 = vsetq_lane_u32(9, qu32, 0);
  assert(qu32[0] == 9 && qu32[1] == 4000000000u);
  int32x2_t ds32 = {-2000000000, 5};
  assert(vget_lane_s32(ds32, 0) == -2000000000);

  uint64x2_t qu64 = {1, 18000000000000000000ull};
  assert(vgetq_lane_u64(qu64, 1) == 18000000000000000000ull);
  int64x1_t ds64 = {-9000000000000000000ll};
  assert(vget_lane_s64(ds64, 0) == -9000000000000000000ll);
  ds64 = vset_lane_s64(-3, ds64, 0);
  assert(ds64[0] == -3);

  float32x4_t qf32 = {1.5f, -2.25f, 3.0e10f, -0.0f};
  assert(vgetq_lane_f32(qf32, 1) == -2.25f);
  qf32 = vsetq_lane_f32(0.5f, qf32, 2);
  assert(qf32[2] == 0.5f && qf32[1] == -2.25f);
  float64x2_t qf64 = {1.5, -2.25};
  assert(vgetq_lane_f64(qf64, 1) == -2.25);
  float64x1_t df64 = vset_lane_f64(4.0, vdup_n_f64(1.0), 0);
  assert(vget_lane_f64(df64, 0) == 4.0);

  float16x4_t df16 = {1, 2, 3, 4};
  assert(vget_lane_f16(df16, 2) == 3);
  df16 = vset_lane_f16(0.5, df16, 0);
  assert(df16[0] == (float16_t)0.5 && df16[1] == 2);

  poly8x16_t qp8 = vdupq_n_p8(3);
  qp8 = vsetq_lane_p8(250, qp8, 4);
  assert(vgetq_lane_p8(qp8, 4) == 250 && vgetq_lane_p8(qp8, 5) == 3);
  poly16x4_t dp16 = {1, 2, 65535, 4};
  assert(vget_lane_p16(dp16, 2) == 65535);
  return 0;
}
