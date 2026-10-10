#include <assert.h>
#include <stdio.h>

char nondet_char();

int main()
{
  char s[4];
  for (int i = 0; i < 3; ++i)
    s[i] = nondet_char();
  s[3] = '\0';

  int x = 7, y = 9, len = 5;
  int r = sscanf(s, "%d %d%n", &x, &y, &len);
  assert(r >= EOF && r <= 2);
  if (r < 2)
    assert(y == 9);
  if (r < 1)
    assert(x == 7);
  return 0;
}
