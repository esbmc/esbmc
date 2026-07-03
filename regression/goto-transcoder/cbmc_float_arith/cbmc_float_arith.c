#include <assert.h>
int main()
{
  float a = 3.5f;
  float b = 1.5f;
  float sum = a + b;
  float diff = a - b;
  float prod = a * b;
  float quot = a / b;
  assert(sum == 5.0f);
  assert(diff == 2.0f);
  assert(prod == 5.25f);
  assert(quot > 2.33f && quot < 2.34f);
  int x = 7;
  int y = 2;
  assert(x / y == 3);
  assert(x % y == 1);
  return 0;
}
