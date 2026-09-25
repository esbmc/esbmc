#include <assert.h>
int main()
{
  double d = __builtin_nan("");
  assert(d == 0.0); // NaN compares equal to nothing, so this must FAIL
  return 0;
}
