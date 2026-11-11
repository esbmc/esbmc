#include <assert.h>
#include <math.h>

/* A tolerance the default 8 terms cannot meet: its sibling control runs the
   same program without the flag and must fail. */
int main()
{
  double y = expm1(0.24);
  assert(fabs(y - 0.27124915032140468) < 1e-15);
  return 0;
}
