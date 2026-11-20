#include <assert.h>
#include <stdio.h>

int main()
{
  int a[2] = {1, 2};
  int r = sscanf("5", "%ld", &a[0]);
  if (r == 1)
    assert(a[1] == 2);
  return 0;
}
