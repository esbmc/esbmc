#include <assert.h>
#include <math.h>

int main()
{
  float x = NAN;
  assert(x != x);  // Should pass: NaN != NaN is true
  return 0;
}
