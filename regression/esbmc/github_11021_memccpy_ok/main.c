#include <assert.h>
#include <string.h>
int main()
{
  char b[8];
  char *e = memccpy(b, "abcd", 'c', 5);
  assert(e == b + 3 && b[2] == 'c');
  assert(memccpy(b, "abcd", 'z', 2) == 0);
  return 0;
}
