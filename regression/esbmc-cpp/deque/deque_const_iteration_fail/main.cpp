#include <cassert>
#include <deque>

static int sum(const std::deque<int> &d)
{
  int s = 0;
  for (int x : d)
    s += x;
  return s;
}

int main()
{
  std::deque<int> d;
  d.push_back(1);
  d.push_back(2);
  assert(sum(d) == 99);
  return 0;
}
