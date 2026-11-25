#include <assert.h>
#include <stdio.h>

int main()
{
  int a[2] = {1, 2};
  void *p = &a[0];
  sscanf("5", "%d", p);
  assert(a[1] == 2);
  return 0;
}
