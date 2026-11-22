#include <assert.h>
#include <stdio.h>

int main()
{
  char b[4] = {'a', 'b', 'c', 'd'};
  sscanf("xyz", "%2c", &b[0]);
  assert(b[2] == 'c');
  assert(b[3] == 'd');
  return 0;
}
