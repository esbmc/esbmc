// #9387: after erasing [begin+1, begin+3) the returned iterator points to 4.
#include <cassert>
#include <deque>

int main()
{
  std::deque<int> d{1, 2, 3, 4};
  std::deque<int>::iterator it = d.erase(d.begin() + 1, d.begin() + 3);
  assert(*it != 4);
  return 0;
}
