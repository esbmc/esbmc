#include <assert.h>
#include <math.h>

int main(void)
{
  assert(fma(2.0, 3.0, 4.0) == 10.0);
  return 0;
}
