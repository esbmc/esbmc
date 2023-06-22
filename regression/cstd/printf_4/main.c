#include <stdio.h>
#include <assert.h>

int main()
{
  char ss[] = "runoob";

  int x = printf("%s", ss);
  assert(x == 6);
  assert(++x == 7);
}