#include "shared.h"
int use_b()
{
  Derived d;
  Base *p = &d;
  return p->f();
}
