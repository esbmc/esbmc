// std::reverse_iterator declared members but defined none, so explicit
// construction was a PARSING ERROR and every member returned nondet (#7535).
#include <cassert>
#include <iterator>

int main()
{
  double a[4] = {1, 2, 3, 4};
  std::reverse_iterator<double *> r(a + 4);
  assert(*r == 4.0);
  ++r;
  assert(*r == 3.0);
  assert(r.base() == a + 3);

  std::reverse_iterator<double *> d;
  d = r;
  assert(*d == 3.0);

  std::reverse_iterator<double *> s = r + 1;
  assert(*s == 2.0);
  assert(r[0] == 3.0);
  return 0;
}
