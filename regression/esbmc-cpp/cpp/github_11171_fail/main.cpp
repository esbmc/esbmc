// #11171: value-initialisation zeroes the object, so a nonzero expectation
// must fail.
#include <cassert>

struct P
{
  int x;
};

int main()
{
  P *h = new P();
  assert(h->x == 1);
  return 0;
}
