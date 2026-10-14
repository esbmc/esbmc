#include <cassert>
#include <deque>

int main()
{
  std::deque<int> d;
  d.push_back(1);
  d.push_back(2);

  d.back() = 42;
  assert(d.back() == 2);
  return 0;
}
