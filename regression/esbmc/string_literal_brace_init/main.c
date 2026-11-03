/* A braced string literal initialises the whole char array (C11 6.7.9p14). */
#include <assert.h>
int main()
{
  char a[4] = {"ab"};
  assert(a[0] == 'a' && a[1] == 'b' && a[2] == 0 && a[3] == 0);
  char b[] = {"xy"};
  assert(sizeof b == 3 && b[1] == 'y');
  return 0;
}
