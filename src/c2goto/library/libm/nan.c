#include <math.h>

double nan(const char *x)
{
  return __builtin_nan(x);
}

float nanf(const char *x)
{
  return __builtin_nanf(x);
}

long double nanl(const char *x)
{
  return __builtin_nanl(x);
}
