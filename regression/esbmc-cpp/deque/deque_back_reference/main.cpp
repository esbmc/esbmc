#include <cassert>
#include <deque>

int main()
{
  std::deque<int> d;
  d.push_back(1);
  d.push_back(2);
  d.push_back(3);

  d.back() = 42;
  d.front() = 7;

  assert(d.back() == 42);
  assert(d.front() == 7);
  assert(d[2] == 42);
  assert(d[0] == 7);
  assert(d[1] == 2);

  int &r = d.back();
  r -= 2;
  assert(d.back() == 40);
  return 0;
}
