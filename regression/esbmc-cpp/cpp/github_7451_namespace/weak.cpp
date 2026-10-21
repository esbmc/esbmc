#include <cassert>

namespace N
{
__attribute__((weak)) int f(int a)
{
  return a + 1;
}
int g __attribute__((weak)) = 1;
} // namespace N

int main()
{
  assert(N::f(3) == 6);
  assert(N::g == 2);
  return 0;
}
