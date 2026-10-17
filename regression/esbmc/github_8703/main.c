/* github #8703: with no digits *endptr is nptr (C11 7.22.1.4p7); on overflow
 * errno is ERANGE and the whole digit run is consumed (p5, p8). */
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
int main(void)
{
  char *end;
  const char *s = "  -x";
  assert(strtol(s, &end, 10) == 0 && end == s);
  const char *h = "0xg";
  assert(strtol(h, &end, 16) == 0 && end == h + 1);
  assert(strtoul(h, &end, 0) == 0 && end == h + 1);

  const char *big = "99999999999999999999";
  errno = 0;
  assert(strtol(big, &end, 10) == LONG_MAX);
  assert(errno == ERANGE && end == big + 20);
  errno = 0;
  assert(strtoll("-99999999999999999999", NULL, 10) == LLONG_MIN);
  assert(errno == ERANGE);
  errno = 0;
  assert(strtoul(big, &end, 10) == ULONG_MAX);
  assert(errno == ERANGE && end == big + 20);

  errno = 0;
  assert(strtoll("-9223372036854775808", NULL, 10) == LLONG_MIN);
  assert(strtol("-12", NULL, 10) == -12);
  assert(errno == 0);
  return 0;
}
