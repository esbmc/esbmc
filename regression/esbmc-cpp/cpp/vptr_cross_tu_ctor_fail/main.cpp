#include "shape.h"
#include <cassert>
int main()
{
  Shape s(7);
  Shape *p = &s;
  // Shape::kind() returns 1, so this assertion must fail.
  assert(p->kind() == 2);
  return 0;
}
