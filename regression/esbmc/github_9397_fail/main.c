#include <assert.h>
#include <stdio.h>

int main(void)
{
  char b[64];
  int r = snprintf(b, sizeof b, "%e|%g", 1.5, 0.0001);
  assert(r != 19);
  return 0;
}
