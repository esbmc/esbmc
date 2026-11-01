#include <assert.h>
#include <ctype.h>
#include <locale.h>
#ifdef __APPLE__
#  include <xlocale.h>
#endif

int main()
{
  locale_t c = newlocale(LC_ALL_MASK, "C", (locale_t)0);
  assert(tolower_l('A', c) == 'A');
  assert(isdigit_l('a', c));
  return 0;
}
