#include <cassert>

extern "C" int nondet_int();

// C++ lets a switch condition be a declaration. The declaration has to be
// hoisted ahead of the switch, which then switches on the declared symbol; left
// in place it *is* the switched value and reaches the solver as a statement.
int main()
{
  int i = nondet_int();
  int n = i < 0 ? 1 : 2;

  switch (int x = n)
  {
  case 1:
    assert(x == 1 && i < 0);
    break;

  case 2:
    assert(x == 1);
    break;

  default:
    assert(0);
    break;
  }
}
