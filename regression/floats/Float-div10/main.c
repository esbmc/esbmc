#include <math.h>
#include <assert.h>

int main()
{
  float a = -INFINITY, b = 0.0f, c;
  c = a/b;
  assert(isinf(c) && c < 0); // Should be -infinity
  return 0;
}
