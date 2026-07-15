#include <assert.h>
int main()
{
  double d = __builtin_huge_val();
  assert(d == 0.0); // +Inf is not 0, so this must FAIL
  return 0;
}
