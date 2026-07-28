// A conditional whose arms are lvalues is an lvalue ([expr.cond]), so binding a
// reference *parameter* to it must designate the selected arm.  The copy
// constructor takes `const P &`, so `P r = c ? a : b;` binds one, and the
// address the frontend synthesises for that binding used to cover neither arm
// (github #6291 fixed only the source-level `&`).
#include <cassert>

struct P
{
  int v;
};

int by_ref(const P &x)
{
  return x.v;
}

int main()
{
  P a{1}, b{2};
  int c = 0;

  P copy_init = (c < 1) ? a : b;
  assert(copy_init.v == 1);

  P assigned;
  assigned = (c < 1) ? a : b;
  assert(assigned.v == 1);

  assert(by_ref((c < 1) ? a : b) == 1);

  return 0;
}
