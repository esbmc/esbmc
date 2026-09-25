#include <list>
#include <iterator>
#include <cassert>

int main()
{
  std::list<double> l;
  l.push_back(1.0);
  l.push_back(2.0);
  l.push_back(3.0);
  std::list<double>::iterator it = l.begin();
  std::advance(it, 2);
  // advance(it, 2) lands on the third element, not the second.
  assert(*it == 2.0);
  return 0;
}
