/* A braced string literal initialises the whole char array (C11 6.7.9p14). */
#include <assert.h>
int main()
{
  char a[4] = {"ab"};
  assert(a[1] == 0);
  return 0;
}
