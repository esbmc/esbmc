#include <cassert>
#include <list>

int main()
{
  std::list<int> l;
  l.push_back(4);
  l.push_back(6);
  int s = 0;
  for (std::list<int>::const_iterator it = l.cbegin(); it != l.cend(); ++it)
    s += *it;
  assert(s == 10);
  return 0;
}
