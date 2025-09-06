#include <math.h>
#include <assert.h>

int main() {
  double x1 = 0.01;
  double x2 = -0.01;
  double approx1 = log1p(x1);
  double approx2 = log1p(x2);
  double true1 = log(1.0 + x1);
  double true2 = log(1.0 + x2);

  // Allow a small tolerance, since it's a truncated series
  assert(fabs(approx1 - true1) < 1e-7);
  assert(fabs(approx2 - true2) < 1e-7);
}

