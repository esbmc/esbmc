// [iterator.synopsis] puts advance and distance in <iterator>, but the
// definitions lived in <algorithm> while <iterator> only declared them. A TU
// that included <iterator> without <algorithm> got a distance() returning an
// unconstrained value and an advance() that did nothing (github #5868).
#include <iterator>
#include <vector>
#include <cassert>

int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  assert(std::distance(v.begin(), v.end()) == 3);
  assert(std::distance(v.begin(), v.begin()) == 0);

  std::vector<int>::iterator i = v.begin();
  std::advance(i, 2);
  assert(*i == 3);
  std::advance(i, -1);
  assert(*i == 2);

  int a[4] = {5, 6, 7, 8};
  assert(std::distance(a, a + 4) == 4);
  return 0;
}
