#include <assert.h>
#include <stdio.h>

int main()
{
  int a[3] = {1, 2, 3};
  sscanf("5", "%d", a);
  assert(a[1] == 2);
  return 0;
}
