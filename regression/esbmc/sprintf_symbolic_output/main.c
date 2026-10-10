#include <assert.h>
#include <stdio.h>

int nondet_int(void);

int main(void)
{
  int x = nondet_int();

  char b[12] = "abcdefghijk";
  int r = sprintf(b, "%d", x);
  assert(r >= 1 && r <= 11);
  assert(b[r] == '\0');
  assert(b[11] == '\0');
  assert(r != 1 || b[2] == 'c');

  char c[8] = "abcdefg";
  int m = snprintf(c, 3, "%u", (unsigned)x);
  assert(m >= 1 && m <= 10);
  assert(c[1] == '\0' || c[2] == '\0');
  assert(c[3] == 'd');
  return 0;
}
