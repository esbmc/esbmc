#include <assert.h>
#include <stdio.h>

int main()
{
  int x, n = -1;
  int r = sscanf("12", "%d%n", &x, &n);
  if (r < 1)
    assert(n == -1);
  if (r == 1)
    assert(n == -1);
  return 0;
}
