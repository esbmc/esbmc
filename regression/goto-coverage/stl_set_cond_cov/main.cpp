#include <set>
#include <cassert>

int main()
{
  std::set<int> s;

  std::pair<std::set<int>::iterator, bool> r1 = s.insert(5);
  if (r1.second)
    s.insert(10);

  std::pair<std::set<int>::iterator, bool> r2 = s.insert(5);
  if (r2.second)
    s.insert(15);

  assert(s.size() == 2);
  return 0;
}
