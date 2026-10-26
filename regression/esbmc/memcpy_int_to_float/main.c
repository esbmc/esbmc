#include <assert.h>
#include <stdint.h>
#include <string.h>

int main()
{
  uint64_t u = 0x4000000000000000;
  double d;
  memcpy(&d, &u, sizeof d);
  assert(d == 2.0);

  uint32_t v = 0xc0400000;
  float f;
  memcpy(&f, &v, sizeof f);
  assert(f == -3.0f);
  return 0;
}
