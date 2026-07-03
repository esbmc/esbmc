#include <assert.h>
#include <math.h>
int main()
{
  float x = 4.0f;
  float y = sqrtf(x);
  assert(y == 3.0f);
  return 0;
}
