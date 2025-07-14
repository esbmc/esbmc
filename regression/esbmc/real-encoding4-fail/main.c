#include <assert.h>
#include <math.h>

int main()
{
  float x = NAN;
  float y = 1.0f;
  assert(x == y);  // Should fail: NaN != 1.0f
  return 0;
}
