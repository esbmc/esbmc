#include <string.h>

int main()
{
  char a[4] = {1, 2, 3, 4};

  /* Reading 8 bytes from a 4-byte object is out of bounds: the intrinsic must
   * report a dereference failure rather than silently scanning past the end. */
  memchr(a, 9, 8);

  return 0;
}
