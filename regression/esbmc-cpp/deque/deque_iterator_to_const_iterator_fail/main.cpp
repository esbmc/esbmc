#include <cassert>
#include <deque>

int main()
{
  std::deque<int> d;
  d.push_back(1);
  d.push_back(2);

  std::deque<int>::const_iterator c = d.begin();
  assert(*c == 2);
  return 0;
}
