#include <assert.h>
#include <stdio.h>

int main(void)
{
  char b[64];
  assert(snprintf(b, sizeof b, "%e", 1.5) == 12);
  assert(snprintf(b, sizeof b, "%E", 0.0) == 12);
  assert(snprintf(b, sizeof b, "%.3e", 2.0) == 9);
  assert(snprintf(b, sizeof b, "%.0e", 2.5) == 5);
  assert(snprintf(b, sizeof b, "%#.0e", 1.0) == 6);
  assert(snprintf(b, sizeof b, "%e", 1e-300) == 13);
  assert(snprintf(b, sizeof b, "%g", 1.5) == 3);
  assert(snprintf(b, sizeof b, "%g", 0.0001) == 6);
  assert(snprintf(b, sizeof b, "%g", 0.00001234) == 9);
  assert(snprintf(b, sizeof b, "%g", 1e6) == 5);
  assert(snprintf(b, sizeof b, "%g", 123456.0) == 6);
  assert(snprintf(b, sizeof b, "%G", 9.9999995) == 2);
  assert(snprintf(b, sizeof b, "%#g", 100.0) == 7);
  assert(snprintf(b, sizeof b, "%.0g", 0.5) == 3);
  assert(snprintf(b, sizeof b, "%g|%e", __builtin_inf(), -__builtin_inf()) == 8);
  assert(snprintf(b, sizeof b, "%e|%G", __builtin_nan(""), __builtin_nan("")) == 7);
  return 0;
}
