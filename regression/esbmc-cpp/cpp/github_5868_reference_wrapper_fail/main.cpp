#include <functional>
#include <cassert>

int main()
{
  int n = 3;
  std::reference_wrapper<int> r = std::ref(n);
  // The wrapper aliases n, so writing through it is visible in n.
  r.get() = 7;
  assert(n == 3);
  return 0;
}
