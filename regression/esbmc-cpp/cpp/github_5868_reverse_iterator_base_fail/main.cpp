#include <list>
#include <cassert>

int main()
{
  std::list<int> l;
  l.push_back(1);
  l.push_back(2);
  std::list<int>::reverse_iterator r = l.rbegin();
  // base() is one step FORWARD of what the reverse iterator denotes, so
  // rbegin().base() is end(), not the last element.
  assert(*r.base() == 2);
  return 0;
}
