// #8008
#include <assert.h>

int main()
{
  int x[2] = {1, 1};
  int y = 1;
  int *p = x;
  int *r = &p[1];
  int a = y + 1;
  *r = 5;
  int b = y + 1;
  assert(b == 2);
  return 0;
}
