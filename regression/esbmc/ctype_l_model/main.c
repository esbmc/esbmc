#include <assert.h>
#include <ctype.h>
#include <locale.h>
#ifdef __APPLE__
#  include <xlocale.h>
#endif

int main()
{
  locale_t c = newlocale(LC_ALL_MASK, "C", (locale_t)0);
  assert(tolower_l('A', c) == 'a' && tolower_l('1', c) == '1');
  assert(toupper_l('b', c) == 'B' && toupper_l('Z', c) == 'Z');
  assert(isalpha_l('x', c) && !isalpha_l('3', c));
  assert(isdigit_l('7', c) && !isdigit_l('a', c));
  assert(isalnum_l('q', c) && !isalnum_l('-', c));
  assert(isspace_l('\t', c) && !isspace_l('x', c));
  assert(isupper_l('Q', c) && islower_l('q', c));
  assert(isxdigit_l('F', c) && !isxdigit_l('g', c));
  assert(ispunct_l('!', c) && isprint_l(' ', c) && !isgraph_l(' ', c));
  assert(iscntrl_l('\n', c) && isblank_l(' ', c));
  return 0;
}
