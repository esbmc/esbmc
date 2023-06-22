#include <stdio.h>
#include <assert.h>

int main()
{
  char *s = "abcde1234151";
  int data = 1000000;
  int x = printf("%s%d\n", s, data);
#ifndef WIN32
  assert(x == 20);
#endif
  x += 1;
}