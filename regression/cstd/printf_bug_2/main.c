#include <stdio.h>
#include <assert.h>
int main()
{
  long s = 1000000000000000000;
  int x = printf("%ld\n", s);
  assert(x == 20);
}