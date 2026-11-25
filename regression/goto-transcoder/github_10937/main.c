/* Built with each __builtin_bitreverse64((unsigned long long)x) rewritten in
 * CBMC's symbol table to a 128-bit bitreverse(x), the irep Kani emits for
 * u128::reverse_bits (C has no 128-bit builtin). */
#include <assert.h>
unsigned __int128 nondet_u128(void);
int main(void)
{
  unsigned __int128 one = 1;
  unsigned __int128 high = __builtin_bitreverse64(one << 64);
  assert(high == one << 63);
  unsigned __int128 x = nondet_u128();
  unsigned __int128 r = __builtin_bitreverse64(x);
  unsigned __int128 back = __builtin_bitreverse64(r);
  assert(back == x);
  return 0;
}
