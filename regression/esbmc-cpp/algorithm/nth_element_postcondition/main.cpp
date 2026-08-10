#include <cassert>
#include <vector>
#include <algorithm>

int main()
{
  std::vector<int> v{3, 1, 2};
  std::nth_element(v.begin(), v.begin() + 1, v.end());
  // Sorted order is {1,2,3}, so the middle element must be 2.
  assert(v[1] == 2);
  // Everything before nth compares no greater; everything after, no less.
  assert(!(v[1] < v[0]));
  assert(!(v[2] < v[1]));

  std::vector<int> w{5, 4, 3, 2, 1};
  std::nth_element(w.begin(), w.begin() + 2, w.end());
  assert(w[2] == 3);

  std::vector<int> d{1, 3, 2};
  std::nth_element(d.begin(), d.begin() + 1, d.end(),
                   [](int a, int b) { return a > b; });
  assert(d[1] == 2);

  return 0;
}
