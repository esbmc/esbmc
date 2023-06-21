#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
int main()
{
  char *s = (char *)malloc(100);
  strncpy(s, "runoob", 6);
  char ss[] = "runoob";

  int x = printf("%s", s);
  int y = printf("%s", ss);
  assert(++x == 7);
  assert(++y == 7);
}