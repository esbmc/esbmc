#include <assert.h>
#include <math.h>
int main()
{
  float x = 1.0f;
  float y = 0.0f;
  float z = x / y;
  assert(!isinf(z));
  return 0;
}
