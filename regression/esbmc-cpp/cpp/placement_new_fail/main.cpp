#include <new>
#include <cassert>

int main()
{
  int x = 0;
  new (&x) int(41);
  assert(x == 0); // must fail: placement new wrote 41 through &x
  return 0;
}
