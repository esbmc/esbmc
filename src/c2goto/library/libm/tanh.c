#include <math.h>

double tanh(double x)
{
__ESBMC_HIDE:;
  double e_pos = exp(x);
  double e_neg = exp(-x);
  return (e_pos - e_neg) / (e_pos + e_neg);
}
