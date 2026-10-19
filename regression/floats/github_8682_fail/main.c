/* github #8682: each assertion contradicts C11 F.10.1.4 */
#include <assert.h>
#include <math.h>
int main(void)
{
  assert(atan2(-0.0, -1.0) > 0.0);
  assert(atan2(0.0, -0.0) == 0.0);
  assert(!signbit(atan2(-0.0, 1.0)));
  assert(!isnan(atan2(NAN, 0.0)));
  assert(isnan(atan2(INFINITY, INFINITY)));
  return 0;
}
