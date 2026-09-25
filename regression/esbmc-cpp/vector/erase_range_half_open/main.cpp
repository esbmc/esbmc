#include <cassert>
#include <vector>

int main()
{
  // Half-open: [begin()+1, begin()+3) removes 2 and 3, keeping 1 and 4.
  std::vector<int> v{1, 2, 3, 4};
  std::vector<int>::iterator it = v.erase(v.begin() + 1, v.begin() + 3);
  assert(v.size() == 2);
  assert(v[0] == 1 && v[1] == 4);
  assert(*it == 4);

  std::vector<int> a{1, 2, 3};
  a.erase(a.begin(), a.begin() + 1);
  assert(a.size() == 2 && a[0] == 2 && a[1] == 3);

  std::vector<int> b{1, 2};
  b.erase(b.begin(), b.end());
  assert(b.empty());

  // An empty range erases nothing.
  std::vector<int> c{1, 2};
  c.erase(c.begin() + 1, c.begin() + 1);
  assert(c.size() == 2 && c[1] == 2);

  std::vector<int> d{1, 2, 3};
  assert(d.max_size() >= d.size());

  return 0;
}
