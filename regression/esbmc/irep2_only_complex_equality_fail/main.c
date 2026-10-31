/* #9362 under --clang-c-irep2-adjust-only: a NaN part is unequal to itself. */
#include <assert.h>
#include <math.h>

int main(void)
{
  double _Complex n = NAN;
  assert(n == n);
  return 0;
}
