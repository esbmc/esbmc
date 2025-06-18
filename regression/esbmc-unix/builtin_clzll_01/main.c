#include <stdint.h>
#include <assert.h>

int main()
{
  uint64_t x;
  int clz;

  x = 0x0000000000000001ULL;
  assert(__builtin_clzll(x) == 63);

  x = 0x0000000000001000ULL; // bit12 = 1
  assert(__builtin_clzll(x) == 51);

  x = 0x00000000000000FFULL;
  assert(__builtin_clzll(x) == 56);

  x = 0x8000000000000000ULL;
  assert(__builtin_clzll(x) == 0);

  return 0;
}
