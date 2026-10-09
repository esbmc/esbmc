#include <assert.h>
#include <stdio.h>

int main(void)
{
  char b[8] = "abcdefg";
  sprintf(b, "%d%s", 42, "x");
  assert(b[0] == '4' && b[1] == '2' && b[2] == 'x' && b[3] == '\0');
  assert(b[4] == 'e');

  char c[8] = "abcdefg";
  int n = snprintf(c, 3, "%c%u", 'q', 123u);
  assert(n == 4);
  assert(c[0] == 'q' && c[1] == '1' && c[2] == '\0' && c[3] == 'd');
  return 0;
}
