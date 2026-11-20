#include <assert.h>
#include <stdio.h>
int x = 0;
int main()
{
  void *p = &x;
  int r = sscanf("0 5", "%p%d", &p, (int *)p);
  if (r == 2)
    assert(x == 0 || p != (void *)0);
  return 0;
}
