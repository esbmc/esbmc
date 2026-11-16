#include <assert.h>
#include <stdio.h>

int main(void)
{
  char b[64];
  int r = snprintf(b, sizeof b, "%e", 1.5);
  assert(r != 12);
  return 0;
}
