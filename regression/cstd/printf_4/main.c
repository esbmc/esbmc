#include <stdio.h>
#include <assert.h>

int main()
{
  char ss[] = "runoob";

  int x = printf("%s", ss);
#ifndef WIN32
  assert(x == 6);
  assert(++x == 7);
#endif
}