// #8008
#include <assert.h>
int main()
{
  int x = 1;
  int *p = &x;
  int **pp = &p;
  int *r = &**pp;
  int a = x + 1;
  *r = 5;
  int b = x + 1;
  assert(b == 2);
  return 0;
}
