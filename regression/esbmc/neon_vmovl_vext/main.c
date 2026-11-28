// vmovl widens each lane and vext takes the lanes from n on of a followed by
// b. The expected values are what the same code gives natively on AArch64.
#include <arm_neon.h>
#include <assert.h>

int main(void)
{
  uint8x8_t u8 = {0, 1, 200, 255, 4, 5, 6, 7};
  uint16x8_t wu8 = vmovl_u8(u8);
  assert(wu8[2] == 200 && wu8[3] == 255);
  int8x8_t s8 = {-128, -1, 100, 127, 0, 5, -6, 7};
  int16x8_t ws8 = vmovl_s8(s8);
  assert(ws8[0] == -128 && ws8[1] == -1 && ws8[6] == -6);
  int16x8_t s16 = {0, -1, 30000, -32768, 32767, 5, -6, 7};
  int32x4_t ws16 = vmovl_high_s16(s16);
  assert(ws16[0] == 32767 && ws16[2] == -6);
  uint32x2_t u32 = {4000000000u, 7};
  uint64x2_t wu32 = vmovl_u32(u32);
  assert(wu32[0] == 4000000000u && wu32[1] == 7);

  uint8x16_t a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  uint8x16_t b = vdupq_n_u8(99);
  uint8x16_t e8 = vextq_u8(a, b, 13);
  assert(e8[0] == 13 && e8[2] == 15 && e8[3] == 99 && e8[15] == 99);
  uint16x4_t c = {1, 2, 3, 4}, d = {5, 6, 7, 8};
  uint16x4_t e16 = vext_u16(c, d, 3);
  assert(e16[0] == 4 && e16[1] == 5 && e16[3] == 7);
  float32x4_t f = {1.5f, 2.5f, 3.5f, 4.5f}, g = {-1.0f, -2.0f, -3.0f, -4.0f};
  float32x4_t ef = vextq_f32(f, g, 1);
  assert(ef[0] == 2.5f && ef[2] == 4.5f && ef[3] == -1.0f);
  uint64x2_t h = {1, 2}, k = {3, 4};
  uint64x2_t e64 = vextq_u64(h, k, 1);
  assert(e64[0] == 2 && e64[1] == 3);
  return 0;
}
