/* Built with each __builtin_bitreverse64((unsigned long long)x) rewritten in
 * CBMC's symbol table to a 128-bit bitreverse(x), the irep Kani emits for
 * u128::reverse_bits (C has no 128-bit builtin). */
#include <assert.h>
int main(void)
{
  unsigned __int128 one = 1;
  unsigned __int128 high = __builtin_bitreverse64(one << 64);
  assert(high != one << 63);
  return 0;
}
