#include <cassert>

extern "C" int nondet_int();

// A by-value struct parameter: paired against `this` instead of its own
// parameter, binds_by_reference matches on the type id alone and takes the
// argument's address, so the call is built with a pointer where a struct is
// expected.
struct A
{
  int x;
  int y;
};

struct T
{
  int x;
  T(A a) : x(a.x + a.y)
  {
  }
};

struct U
{
  T t;
  U(A a) : t(a)
  {
  }
};

int main()
{
  A a;
  a.x = nondet_int();
  a.y = 1;
  U u(a);
  assert(u.t.x == a.x + 2);
}
