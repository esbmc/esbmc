#include <assert.h>
extern float fabsf(float);
int main()
{
  float x = -2.5f;
  float y = fabsf(x);
  assert(y == 3.0f);
  return 0;
}
