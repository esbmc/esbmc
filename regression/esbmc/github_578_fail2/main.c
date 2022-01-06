#include <stdint.h>
#include <assert.h>
int main()
{
  uint32_t x = 2;
  x *= 1U << 31;
  x *= 1U << 31;
  assert(x == 1ULL << 63);
}
