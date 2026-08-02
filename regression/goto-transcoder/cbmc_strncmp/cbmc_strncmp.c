#include <assert.h>
#include <string.h>
int main()
{
  char a[4] = "abc", b[4] = "abd";
  assert(strncmp(a, b, 2) == 0); // first two bytes equal
  assert(strncmp(a, b, 3) < 0);  // differ at index 2
  return 0;
}
