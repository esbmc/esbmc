// #8679: the variadic arguments of a call through a member pointer reach the
// callee.
#include <cassert>
#include <cstdarg>

struct B
{
  virtual int f(int n, ...)
  {
    va_list ap;
    va_start(ap, n);
    int v = va_arg(ap, int);
    va_end(ap);
    return v;
  }
};

int main()
{
  B b;
  B *pb = &b;
  int (B::*pf)(int, ...) = &B::f;
  int r = (pb->*pf)(1, 42);
  assert(r != 42);
  return 0;
}
