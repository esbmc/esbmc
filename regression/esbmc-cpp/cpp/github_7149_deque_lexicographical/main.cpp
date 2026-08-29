#include <deque>
#include <cassert>
int main()
{
  std::deque<int> a;
  a.push_back(2);
  std::deque<int> b;
  b.push_back(1);
  b.push_back(3);
  assert(!(a <= b));
  assert(a > b);
  assert(!(a < b));
  assert(b < a);
  assert(a >= b);
  return 0;
}
