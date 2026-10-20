#include <assert.h>
#include <stdio.h>
#include <string.h>

int nondet_int(void);

int main(void)
{
  assert(printf("%+d", 5) == 2);
  assert(printf("% d", 5) == 2);
  assert(printf("%#x %#o", 255, 8) == 8);
  assert(printf("%.3d|%.0d", 5, 0) == 4);
  assert(printf("%5s|%.1s", "ab", "ab") == 7);
  assert(printf("%+.1f|%-5.1f|", 1.5, -2.0) == 11);

  char b[16];
  int n = sprintf(b, "%-4d|%+.3i|%#X", 7, 42, 171);
  assert(n == 14);
  assert(strcmp(b, "7   |+042|0XAB") == 0);

  int x = nondet_int();
  int r = printf("%+d", x);
  assert(r >= 2 && r <= 11);

  char s[8];
  r = printf("%6.3s", s);
  assert(r == 6);
  return 0;
}
