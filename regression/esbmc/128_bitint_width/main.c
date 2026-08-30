#include <assert.h>

int main()
{
  unsigned _BitInt(80) m = 0;
  m = ~m; /* 2^80 - 1 */
  /* Unsigned wraparound is defined, and it happens at 80 bits. Ranking a
     65..127-bit operand INT128 widens both sides and the sum becomes 2^80. */
  assert((unsigned _BitInt(80))(m + 1uwb) == 0uwb);
  return 0;
}
