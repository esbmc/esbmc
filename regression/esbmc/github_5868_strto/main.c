/* strtoul, strtoull, strtod and strtold were declared in stdlib.h but never
 * defined, so they returned an unconstrained value (github #5868). */
#include <stdlib.h>
#include <limits.h>
#include <assert.h>

int main()
{
  char *e;

  assert(strtoul("42", &e, 10) == 42UL);
  assert(*e == 0);
  assert(strtoul("ff", &e, 16) == 255UL);
  assert(strtoul("0x1f", &e, 0) == 31UL);
  /* C99 7.22.1.4p5: a leading '-' is negated in the return type. */
  assert(strtoul("-1", &e, 10) == ULONG_MAX);

  assert(strtoull("42", &e, 10) == 42ULL);
  assert(strtoull("18446744073709551615", &e, 10) == ULLONG_MAX);

  assert(strtod("2.5", &e) == 2.5);
  assert(*e == 0);
  assert(strtod("-4.25", &e) == -4.25);
  assert(strtod("  7", &e) == 7.0);

  assert(strtold("3", &e) == 3.0L);

  /* endptr points at the first unconverted character. */
  assert(strtoul("42abc", &e, 10) == 42UL);
  assert(*e == 'a');
  assert(strtod("2.5xyz", &e) == 2.5);
  assert(*e == 'x');
  return 0;
}
