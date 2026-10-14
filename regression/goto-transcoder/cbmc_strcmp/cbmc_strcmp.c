#include <assert.h>
#include <string.h>
int main()
{
  char a[4] = "abc", b[4] = "abc", c[4] = "abd";
  assert(strcmp(a, b) == 0); // identical
  assert(strcmp(a, c) < 0);  // 'c' < 'd'
  assert(strcmp(c, a) > 0);  // reversed
  return 0;
}
