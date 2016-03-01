#include <uchar.h>
#include <wchar.h>

int main ()
{
  char x = '\n';
  char y = '\0';
  char z = '\1';
  char o = '\144';
  assert (x == 10);
  assert (y == 0);
  assert (z == 1);
  assert (o == 100);

  char16_t a = u'貓';

  char32_t b1 = U'貓';
  char32_t b2 = U'🍌';

  wchar_t c1 = L'β';
  wchar_t c2 = L'貓';

  char d = 'AB';

}
