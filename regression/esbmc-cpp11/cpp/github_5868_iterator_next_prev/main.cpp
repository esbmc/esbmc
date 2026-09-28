// std::next and std::prev ([iterator.operations]/3,6) were missing from the
// <iterator> model entirely (github #5868).
#include <iterator>
#include <vector>
#include <cassert>

int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  assert(*std::next(v.begin()) == 2);
  assert(*std::next(v.begin(), 2) == 3);
  assert(std::next(v.begin(), 3) == v.end());
  assert(*std::prev(v.end()) == 3);
  assert(*std::prev(v.end(), 2) == 2);

  // next/prev take the iterator by value: the argument must not move.
  std::vector<int>::iterator i = v.begin();
  (void)std::next(i, 2);
  assert(*i == 1);

  int a[4] = {5, 6, 7, 8};
  assert(*std::next(a, 3) == 8);
  return 0;
}
