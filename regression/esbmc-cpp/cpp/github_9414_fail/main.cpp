#include <cassert>
#include <functional>
#include <set>

int main()
{
  std::set<int, std::greater<int> > s;
  s.insert(1);
  s.insert(3);
  s.insert(2);
  assert(!(*s.begin() == 3));
  return 0;
}
