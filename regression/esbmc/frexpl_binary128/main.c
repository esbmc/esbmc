#include <assert.h>
#include <math.h>
#include <float.h>

int main()
{
  int e;
  long double m = frexpl(1024.0L, &e);
  assert(m == 0.5L && e == 11);
  m = frexpl(-3.0L, &e);
  assert(m == -0.75L && e == 2);
  m = frexpl(LDBL_MIN / 4, &e);
  assert(m == 0.5L && e == LDBL_MIN_EXP - 2);
  assert(ldexpl(1.0L, 3) == 8.0L);
  assert(ldexpl(0.75L, -2) == 0.1875L);
  assert(scalbnl(3.0L, 4) == 48.0L);
  assert(ilogbl(1024.0L) == 10);
  assert(logbl(0.75L) == -1.0L);
  return 0;
}
