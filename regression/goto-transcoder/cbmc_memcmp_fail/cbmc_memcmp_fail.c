#include <assert.h>
#include <string.h>
int main()
{
  char a[4] = "abc", b[4] = "abd";
  assert(memcmp(a, b, 3) == 0); // differ at index 2, so not equal
  return 0;
}
