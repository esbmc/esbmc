// #9387: deque::erase(first, last) erases [first, last) and returns an
// iterator to the element last pointed to.
#include <cassert>
#include <deque>

int main()
{
  std::deque<int> d{1, 2, 3, 4};
  d.erase(d.begin(), d.begin() + 2);
  assert(d.size() == 2 && d[0] == 3 && d[1] == 4);

  std::deque<int> e{1, 2, 3, 4};
  std::deque<int>::iterator it = e.erase(e.begin() + 1, e.begin() + 3);
  assert(e.size() == 2 && e[0] == 1 && e[1] == 4 && *it == 4);

  it = e.erase(e.begin() + 1, e.end());
  assert(e.size() == 1 && e[0] == 1 && it == e.end());

  it = e.erase(e.begin(), e.begin());
  assert(e.size() == 1 && it == e.begin());
  return 0;
}
