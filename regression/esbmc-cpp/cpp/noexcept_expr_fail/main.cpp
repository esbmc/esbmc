#include <cassert>

void f() noexcept
{
}

int main()
{
  bool b = noexcept(f()); // true: f is noexcept
  assert(b == false);     // should fail: b is true
  return 0;
}
