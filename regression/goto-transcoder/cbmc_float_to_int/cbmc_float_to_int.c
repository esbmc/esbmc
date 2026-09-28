#include <stdint.h>
int main()
{
  float f = 3.75f;
  double d = -2.5;
  int8_t a = (int8_t)f;
  int32_t b = (int32_t)f;
  int64_t c = (int64_t)d;
  uint32_t u = (uint32_t)f;
  __CPROVER_assert(a == 3, "f32 -> i8");
  __CPROVER_assert(b == 3, "f32 -> i32");
  __CPROVER_assert(c == -2, "f64 -> i64");
  __CPROVER_assert(u == 3u, "f32 -> u32");
  return 0;
}
