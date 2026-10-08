#include <assert.h>
#include <fenv.h>

int main()
{
  long long x = 1LL << 53;
  fesetround(FE_UPWARD);
  x += 1.0;
  assert(x == (1LL << 53));
}
