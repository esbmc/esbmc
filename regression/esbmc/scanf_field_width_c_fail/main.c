#include <assert.h>
#include <stdio.h>

int main()
{
  char b[4] = {'a', 'b', 'c', 'd'};
  int r = sscanf("xyz", "%3c", &b[0]);
  if (r == 1)
    assert(b[1] == 'b');
  return 0;
}
