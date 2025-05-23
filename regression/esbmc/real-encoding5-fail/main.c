#include <assert.h>
#include <math.h>

int main()
{
  float x = NAN;
  float y = NAN;
  _Bool z = nondet_bool();

  if (z) 
    assert(x == y); // Should fail: NaN != NaN even if both are NaN

  assert (z == 0 || z == 1);
  return 0;
}
