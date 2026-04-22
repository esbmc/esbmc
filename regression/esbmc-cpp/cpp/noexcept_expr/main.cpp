#include <cassert>

void f() noexcept
{
}
void g()
{
}

int main()
{
  bool b1 = noexcept(f()); // true: f is noexcept
  bool b2 = noexcept(g()); // false: g may throw
  assert(b1 == true);
  assert(b2 == false);
  return 0;
}
