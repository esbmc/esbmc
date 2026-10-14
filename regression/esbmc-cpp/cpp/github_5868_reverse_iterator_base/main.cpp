#include <list>
#include <cassert>

int main()
{
  std::list<int> l;
  l.push_back(1);
  l.push_back(2);
  l.push_back(3);

  // [reverse.iter.conv]: &*rit == &*(rit.base() - 1), so rbegin().base() is
  // end() and rend().base() is begin().
  std::list<int>::reverse_iterator r = l.rbegin();
  assert(*r == 3);
  assert(r.base() == l.end());

  std::list<int>::reverse_iterator e = l.rend();
  assert(e.base() == l.begin());
  assert(*e.base() == 1);

  ++r;
  assert(*r == 2);
  assert(*r.base() == 3);
  return 0;
}
