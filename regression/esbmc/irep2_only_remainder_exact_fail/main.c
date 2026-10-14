#include <assert.h>
#include <math.h>

int main(void)
{
  assert(!signbit(remainder(-1.0, 1.0)));
  return 0;
}
