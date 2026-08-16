#include <assert.h>

int main()
{
  int a, b, c;
  int *p = 0;

  assert(a + b == b + a);
  assert(a + b == a + c);
  *p = 1;
  return 0;
}
