#include <stdio.h>
#include <assert.h>

int main()
{
  char ss[] = "runoob";
  char* fmt = "%s";

  int x = printf(fmt, ss);
#ifndef _WIN32
  assert(x == 6);
  assert(++x == 7);
#endif
}