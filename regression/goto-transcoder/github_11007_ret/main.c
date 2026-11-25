#include <assert.h>
#include <string.h>
int main()
{
  char a[4] = {1, 2, 3, 4}, b[8];
  assert(memcpy(b, a, 4) == b);
  assert(memmove(a + 1, a, 3) == a + 1);
  assert(memset(b, 7, 3) == b);
  assert(strcpy(b, "ab") == b);
  assert(strcat(b, "cd") == b);
  assert(strncat(b, "ef", 1) == b);
  assert(strncpy(a, "x", 4) == a);
  return 0;
}
