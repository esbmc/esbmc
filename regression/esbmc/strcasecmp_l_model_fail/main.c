#include <assert.h>
#include <locale.h>
#include <strings.h>
#ifdef __APPLE__
#  include <xlocale.h>
#endif

int main()
{
  locale_t c = newlocale(LC_ALL_MASK, "C", (locale_t)0);
  assert(strcasecmp_l("b", "A", c) < 0);
  assert(strncasecmp_l("AbX", "abY", 3, c) == 0);
  return 0;
}
