#include <deque>
#include <cassert>

int main()
{
  std::deque<int> a;
  a.push_back(2);
  std::deque<int> b;
  b.push_back(1);
  b.push_back(3);
  // Ordering by size instead of lexicographically would make this hold.
  assert(a <= b);
  return 0;
}
