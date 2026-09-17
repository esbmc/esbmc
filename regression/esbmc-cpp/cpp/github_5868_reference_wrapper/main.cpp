#include <functional>
#include <cassert>
int main()
{
  int n = 3;
  std::reference_wrapper<int> r = std::ref(n);
  r.get() = 7;
  assert(n == 7);

  int m = r;
  assert(m == 7);

  std::reference_wrapper<const int> c = std::cref(n);
  assert(c.get() == 7);
  n = 9;
  assert(c.get() == 9);
  return 0;
}
