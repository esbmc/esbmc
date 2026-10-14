#include <assert.h>

long double fmal(double, double, double);

int main(void)
{
  assert(fmal(2.0, 3.0, 4.0) == 10.0L);
  return 0;
}
