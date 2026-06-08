#include <cassert>

// A function with a restrictive (empty dynamic) exception specification may
// throw and catch an otherwise-disallowed exception *internally*: the
// specification is only violated when handler search exits the function body.
// Here the int never crosses f's boundary, so there is no violation.
void f() throw()
{
  int caught = 0;
  try
  {
    throw 5;
  }
  catch (int)
  {
    caught = 1;
  }
  assert(caught == 1);
}

int main()
{
  f();
  return 0;
}
