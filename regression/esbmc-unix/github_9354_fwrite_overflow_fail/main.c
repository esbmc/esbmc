/* #9354: size * nitems wraps size_t, so no object holds the bytes read. */
#include <stdio.h>

int main(void)
{
  char b[4] = {'a', 'b', 'c', 'd'};
  fwrite(b, (1ULL << 63) + 1, 2, stdout);
  return 0;
}
