#include <assert.h>
#include <stdio.h>

int main()
{
  char buf[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
  int r = sscanf("xy", "%s", &buf[2]);
  if (r == 1)
    assert(buf[3] == 'd');
  return 0;
}
