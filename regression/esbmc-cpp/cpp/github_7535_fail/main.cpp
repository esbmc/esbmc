// A reverse iterator refers to the element before its base ([reverse.iter.elem]),
// so *reverse_iterator(a + 4) is a[3], not a[0] (#7535).
#include <cassert>
#include <iterator>

int main()
{
  double a[4] = {1, 2, 3, 4};
  std::reverse_iterator<double *> r(a + 4);
  assert(*r == 1.0);
  return 0;
}
