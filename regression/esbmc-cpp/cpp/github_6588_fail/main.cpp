#include <cassert>

int main()
{
  // [dcl.init]/7: the non-parenthesised form default-initialises, which leaves
  // a scalar element indeterminate -- the zero must not be assumed here.
  int *p = new int[4];
  assert(p[0] == 0);
  delete[] p;
  return 0;
}
