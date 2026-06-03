#include <stdint.h>
#include <assert.h>

int main()
{
  uint64_t x;

  // __builtin_clzll(0) is undefined in C. ESBMC now lowers clz to a value
  // (width - popcount of the right-smeared argument), which models the zero
  // case as the bit width rather than flagging it. See #4606.
  x = 0;
  assert(__builtin_clzll(x) == 64);

  return 0;
}
