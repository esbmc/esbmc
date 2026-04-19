#include <assert.h>

int main()
{
  float x = 1e-154, y = 1e-154;
  float result = x * y;
  assert(result > 0.0);
  return 0;
}
