#include <assert.h>
#include <math.h>

int main()
{
  double r = log(2.0);
  assert(r > 0.69314 && r < 0.69315);
  assert(log(M_E) != 1.0);
  return 0;
}
