#include <assert.h>
#include <float.h>

int main() {
  double x = DBL_TRUE_MIN; // Smallest positive subnormal double
  double y = 0.5;
  double z = x * y;

  assert(z > 0.0); // Fails: gradual underflow
  return 0;
}

