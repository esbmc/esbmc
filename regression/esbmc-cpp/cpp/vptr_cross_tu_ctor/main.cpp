#include "shape.h"
#include <cassert>
int main()
{
  Shape s(7);
  Shape *p = &s;
  assert(p->kind() == 1); // base object dispatches to Shape::kind

  Circle c(9);
  Circle *q = &c;
  assert(q->kind() == 2); // derived object dispatches to Circle::kind
  return 0;
}
