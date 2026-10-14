#include <assert.h>
int main()
{
  int x = 5;
  int *p;
  p = &x;
  *p = 10;
  assert(x == 10);
  return 0;
}
