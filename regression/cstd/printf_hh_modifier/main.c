#include <assert.h>
#include <stdio.h>

int main()
{
  /* %hhd truncates to signed char: 256 & 0xFF = 0 → 1 char */
  int x = printf("%hhd", 256);
#ifndef _WIN32
  assert(x == 1);
#endif
}
