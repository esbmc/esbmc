#include "shared.h"
int use_a()
{
  Derived d;
  Base *p = &d;
  return p->f();
}
