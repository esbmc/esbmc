#include <assert.h>
#include <math.h>

int main()
{
  double r = log10(2.0);
  assert(r > 0.30102 && r < 0.30103);
  assert(log10(1000.0) != 3.0);
  return 0;
}
