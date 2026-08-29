#include <assert.h>
int main(void)
{
  long double frac = 3.75L;
  assert((int)frac == 4); /* truncation gives 3, so this must fail */
  return 0;
}
