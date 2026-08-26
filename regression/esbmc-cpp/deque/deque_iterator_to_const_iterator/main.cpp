#include <cassert>
#include <deque>

int main()
{
  std::deque<int> d;
  d.push_back(1);
  d.push_back(2);
  d.push_back(3);

  int s = 0;
  for (std::deque<int>::const_iterator it = d.begin(); it != d.end(); ++it)
    s += *it;
  assert(s == 6);

  std::deque<int>::const_iterator c = d.begin();
  assert(*c == 1);
  return 0;
}
