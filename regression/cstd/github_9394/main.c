// #9394: an int argument supplies a '*' field width or precision
// (C11 7.21.6.1p5).
#include <assert.h>
#include <stdio.h>

int main(void)
{
  char b[16];

  int r = snprintf(b, sizeof b, "%*d", 6, 42);
  assert(r == 6);
  assert(b[0] == ' ' && b[4] == '4' && b[5] == '2');

  r = sprintf(b, "%*d", -3, 7);
  assert(r == 3);
  assert(b[0] == '7' && b[2] == ' ');

  r = snprintf(b, sizeof b, "%.*d", 4, 7);
  assert(r == 4);
  assert(b[0] == '0' && b[3] == '7');

  r = snprintf(b, sizeof b, "%.*d", -1, 7);
  assert(r == 1);

  r = snprintf(b, sizeof b, "%*s|", 5, "ab");
  assert(r == 6);
  assert(b[3] == 'a' && b[5] == '|');

  r = snprintf(b, sizeof b, "%-*.*s|%d", 4, 1, "ab", 9);
  assert(r == 6);
  assert(b[0] == 'a' && b[1] == ' ' && b[4] == '|' && b[5] == '9');
  return 0;
}
