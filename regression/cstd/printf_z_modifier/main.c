#include <assert.h>
#include <stddef.h>
#include <stdio.h>

int main()
{
  /* %zu uses size_t (config-derived width) */
  int x = printf("%zu", (size_t)12345);
#ifndef _WIN32
  assert(x == 5);
#endif
}
