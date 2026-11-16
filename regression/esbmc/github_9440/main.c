#include <assert.h>
#include <stdio.h>

int main(void)
{
  char c = (char)0xe9;
  int r = putchar(c);
  assert(r == 233 || r == EOF);
  r = putchar(300);
  assert(r == 44 || r == EOF);
  return 0;
}
