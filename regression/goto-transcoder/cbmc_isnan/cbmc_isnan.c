#include <assert.h>
#include <math.h>
int main()
{
  float nanval = 0.0f / 0.0f;
  assert(isnan(nanval));
  return 0;
}
