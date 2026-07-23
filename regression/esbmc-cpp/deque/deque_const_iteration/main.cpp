#include <cassert>
#include <deque>

static int sum(const std::deque<int> &d)
{
  int s = 0;
  for (int x : d)
    s += x;
  return s;
}

static int sum_it(const std::deque<int> &d)
{
  int s = 0;
  for (std::deque<int>::const_iterator it = d.begin(); it != d.end(); ++it)
    s += *it;
  return s;
}

int main()
{
  std::deque<int> d;
  d.push_back(1);
  d.push_back(2);
  d.push_back(3);
  assert(sum(d) == 6);
  assert(sum_it(d) == 6);
  return 0;
}
