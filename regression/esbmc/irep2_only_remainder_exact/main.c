#include <assert.h>
#include <math.h>

/* C17 F.10.7.2: remainder returns a zero with the sign of x. */
int main(void)
{
  assert(signbit(remainder(-1.0, 1.0)));
  return 0;
}
