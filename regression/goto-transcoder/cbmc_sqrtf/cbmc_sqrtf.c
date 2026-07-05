#include <assert.h>
#include <math.h>
int main()
{
  float x = 2.5f;
  float y = sqrtf(x);
  assert(y >= 0.0f || y < 0.0f);
  return 0;
}
