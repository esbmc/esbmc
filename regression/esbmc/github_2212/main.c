#include <assert.h>

int main()
{
  _Float16 a = 1.5;
  _Float16 b = 2.5;
  _Float16 result = a + b;

  assert(result == 4.0);

  return 0;
}
