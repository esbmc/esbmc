#include <stdio.h>
#include <assert.h>
int main()
{
  char *s = "abcde123415";
  int x = printf("%s\n", s);
#ifndef _WIN32
  assert(x == 12);
#endif
  x+=1;
}