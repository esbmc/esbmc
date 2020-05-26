#include<math.h>

int main()
{
  double d, q, r;
  __ESBMC_assume(isfinite(q));
  d=q;
  r=d+0;
  assert(r==d);
}
