#include <assert.h>
#include <stdio.h>

int main()
{
  int x = 7, y = 9, n = -1, m = -1;
  int r = sscanf("abc", "%d%n %d%n", &x, &n, &y, &m);
  if (r < 1)
    assert(n == -1);
  if (r < 2)
    assert(m == -1);
  return 0;
}
