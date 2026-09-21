/* github #7896: _Float16 was built with a 4-bit exponent, capping the format
 * at ~255.94, so 1000.0 folded to +inf. IEEE 754 binary16 holds it exactly. */
#include <assert.h>

int main(void)
{
  _Float16 x = (_Float16)1000.0f;
  assert(x == 1000.0f);
  return 0;
}
