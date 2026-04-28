#include <assert.h>
#include <stdio.h>

int main()
{
  /* "%05d" with -3 → "-0003" (5 chars, not "000-3") */
  int x = printf("%05d", -3);
#ifndef _WIN32
  assert(x == 5);
#endif
}
