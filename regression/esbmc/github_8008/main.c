// #8008
#include <assert.h>

int main()
{
  int x[2] = {1, 1};
  int *p = x;
  int *r = &p[1];
  int a = x[1] + 1;
  *r = 5;
  int b = x[1] + 1;
  assert(b == 2); // fails: b == 6
  return 0;
}
