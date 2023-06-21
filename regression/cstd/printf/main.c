#include <stdio.h>
#include <assert.h>
int main()
{
  char *s = "abcde123415";
  int x = printf("%s\n", s);
  assert(x == 12);
}