// The NULL alternative of malloc(0) must be a genuine NULL pointer, not some
// distinct zero-sized object that merely compares equal to it.
#include <stdlib.h>

int main(void)
{
  char *p = malloc(0);
  if (p == 0)
    *p = 1;
  return 0;
}
