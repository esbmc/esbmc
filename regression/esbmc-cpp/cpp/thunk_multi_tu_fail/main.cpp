#include "shared.h"
#include <cassert>
int main()
{
  // Derived::f() returns 2 via virtual dispatch, so this must fail.
  assert(use_a() == 1);
  return 0;
}
