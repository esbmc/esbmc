#include <assert.h>
#include <math.h>

int main()
{
  int e;
  long double m = frexpl(1024.0L, &e);
  assert(m == 0.5L && e == 11);
  assert(ldexpl(1.0L, 3) != 8.0L);
  return 0;
}
