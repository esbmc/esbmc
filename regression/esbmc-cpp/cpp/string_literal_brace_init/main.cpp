// A braced string literal initialises the whole char array ([dcl.init.string]).
#include <cassert>
int main()
{
  char a[4]{"ab"};
  assert(a[0] == 'a' && a[1] == 'b' && a[3] == 0);
  return 0;
}
