#include <assert.h>
#include <stdio.h>

int main()
{
  int x;
  int r = sscanf("abc", "%d", &x);
  assert(r == 1);
  return 0;
}
