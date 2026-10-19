/* github #8702: strtod rounded at every fractional digit, so strtod("0.3")
 * was 0.30000000000000004, and atof had no model. */
#include <assert.h>
#include <stdlib.h>

int main(void)
{
  char *e;
  assert(strtod("0.3", NULL) == 0.3);
  assert(atof("0.7") == 0.7);
  assert(strtof("0.3", NULL) == 0.3f);
  assert(strtold("2.5e-1", NULL) == 0.25L);
  assert(strtod("-123.456", &e) == -123.456);
  assert(*e == 0);
  assert(strtod("1.5e-3x", &e) == 1.5e-3);
  assert(*e == 'x');
  assert(strtod("25E2", NULL) == 2500.0);
  assert(strtod("0e400", NULL) == 0.0);

  /* Past the exact range the result is approximate. */
  double big = strtod("12345678901234567890123", NULL);
  assert(big > 1.2e22 && big < 1.3e22);
  double digits = strtod("0.12345678901234567890123", NULL);
  assert(digits > 0.12 && digits < 0.13);
  assert(strtod("1e-1000000", NULL) == 0.0);
  return 0;
}
