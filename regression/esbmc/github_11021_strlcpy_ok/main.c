#include <assert.h>
#include <string.h>
int main()
{
  char b[4];
  size_t n = strlcpy(b, "ab", 4);
  assert(n == 2 && b[1] == 'b' && b[2] == 0);
  char c[3];
  n = strlcpy(c, "abcdef", 3);
  assert(n == 6 && c[1] == 'b' && c[2] == 0);
  return 0;
}
