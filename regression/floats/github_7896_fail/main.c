/* github #7896: 65504 is the largest finite binary16. Under the 4-bit
 * exponent it overflowed to +inf and this assertion held vacuously. */
#include <assert.h>

int main(void)
{
  _Float16 x = (_Float16)65504.0f;
  assert(x > 65504.0f);
  return 0;
}
