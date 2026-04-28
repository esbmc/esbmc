#include <assert.h>
#include <stdio.h>

int main()
{
  /* "%05d" with 3 → "00003" (5 chars) */
  int x = printf("%05d", 3);
#ifndef _WIN32
  assert(x == 5);
#endif
}
