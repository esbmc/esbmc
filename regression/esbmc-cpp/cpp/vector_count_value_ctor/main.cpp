#include <vector>
#include <cassert>

int main()
{
  // [sequence.reqmts]: with both arguments integral, the (count, value)
  // constructor must win. The iterator-pair template must not participate.
  std::vector<double> d(3, 0);
  assert(d.size() == 3);
  assert(d[0] == 0.0);

  std::vector<long long> l(3, 7);
  assert(l.size() == 3);
  assert(l[2] == 7);

  std::vector<unsigned> u(2, 5);
  assert(u[1] == 5u);

  // size_type is int here, so this one already worked -- keep it pinned.
  std::vector<int> i(4, 9);
  assert(i.size() == 4);
  assert(i[3] == 9);

  // The iterator-pair constructor must still be selected for real iterators.
  int a[3] = {1, 2, 3};
  std::vector<int> v(a, a + 3);
  assert(v.size() == 3);
  assert(v[2] == 3);
  return 0;
}
