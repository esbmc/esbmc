// The temporary a comma's right operand would have been copied out of never
// exists ([expr.comma]/1), so no destructor runs before the variable's own.
#include <cassert>

int dtors = 0;

struct A
{
  int v;
  A(int x) : v(x)
  {
  }
  A(const A &) = delete;
  ~A()
  {
    ++dtors;
  }
};

int main()
{
  int k = 0;
  A a = (++k, A(3));
  assert(dtors == 1);
  return 0;
}
