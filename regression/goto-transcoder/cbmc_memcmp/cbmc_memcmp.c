#include <assert.h>
#include <string.h>
int main()
{
  char a[4] = "abc", b[4] = "abc", c[4] = "abd";
  assert(memcmp(a, b, 4) == 0); // identical buffers compare equal
  assert(memcmp(a, c, 3) < 0);  // 'c' < 'd' -> negative
  assert(memcmp(c, a, 3) > 0);  // reversed -> positive
  return 0;
}
