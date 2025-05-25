#include <math.h>
#include <assert.h>

int main()
{
  float a = 0.0f, b = 0.0f, c;
  c = a/b;
  assert(isnan(c)); // Should be NaN
  return 0;
}
