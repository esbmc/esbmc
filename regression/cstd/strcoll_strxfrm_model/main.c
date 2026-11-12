#include <assert.h>
#include <locale.h>
#include <string.h>

int main(void)
{
  assert(strcoll("abc", "abc") == 0);
  assert(strcoll("abc", "abd") < 0);
  assert(strcoll("b", "abc") > 0);
  assert(strcoll("ab", "abc") < 0);

  assert(strxfrm(NULL, "hello", 0) == 5);
  char out[6];
  assert(strxfrm(out, "hello", sizeof out) == 5);
  assert(strcmp(out, "hello") == 0);
  char small[3];
  assert(strxfrm(small, "hello", sizeof small) == 5);

  locale_t c = newlocale(LC_ALL_MASK, "C", 0);
  assert(strcoll_l("x", "y", c) < 0);
  char o[4];
  assert(strxfrm_l(o, "abc", sizeof o, c) == 3);
  assert(strcmp(o, "abc") == 0);
  return 0;
}
