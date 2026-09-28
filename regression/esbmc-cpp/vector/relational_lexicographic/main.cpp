#include <cassert>
#include <vector>

int main()
{
  // The first differing element decides, whatever the lengths.
  std::vector<int> a{2}, b{1, 3};
  assert(a > b);
  assert(a >= b);
  assert(!(a < b));
  assert(!(a <= b));

  // A prefix is less than the range that extends it.
  std::vector<int> p{1, 2}, q{1, 2, 3};
  assert(p < q);
  assert(p <= q);
  assert(q > p);
  assert(!(p > q));

  // Equal ranges compare equal on both non-strict orderings.
  std::vector<int> e{1, 2}, f{1, 2};
  assert(e <= f);
  assert(e >= f);
  assert(!(e < f));
  assert(!(e > f));

  // Empty is less than anything non-empty.
  std::vector<int> z, o{0};
  assert(z < o);
  assert(!(o < z));

  std::vector<int> c{1, 2}, d{1, 3};
  assert(c < d);
  assert(d > c);

  return 0;
}
