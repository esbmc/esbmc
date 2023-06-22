#include <stdio.h>
#include <assert.h>

int main()
{
  char *s = "abcde1234151";
  int data = 1000000;
  int x = printf("%s%d\n", s, data);
  assert(x == 20);
  x += 1;
}