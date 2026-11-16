#include <assert.h>
#include <stdio.h>

int main(void)
{
  int r = putchar(300);
  assert(r != 44);
  return 0;
}
