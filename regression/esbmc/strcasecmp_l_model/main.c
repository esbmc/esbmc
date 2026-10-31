#include <assert.h>
#include <locale.h>
#include <strings.h>
#ifdef __APPLE__
#  include <xlocale.h>
#endif

int main()
{
  locale_t c = newlocale(LC_ALL_MASK, "C", (locale_t)0);
  assert(strcasecmp_l("AbC", "abc", c) == 0);
  assert(strcasecmp_l("a", "B", c) < 0);
  assert(strcasecmp_l("abcd", "ABC", c) > 0);
  assert(strncasecmp_l("AbX", "abY", 2, c) == 0);
  assert(strncasecmp_l("AbX", "abY", 3, c) < 0);
  assert(strncasecmp_l("x", "y", 0, c) == 0);
  return 0;
}
