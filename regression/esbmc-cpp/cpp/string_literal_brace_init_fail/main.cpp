// A braced string literal initialises the whole char array ([dcl.init.string]).
#include <cassert>
int main()
{
  char a[4]{"ab"};
  assert(a[1] == 0);
  return 0;
}
