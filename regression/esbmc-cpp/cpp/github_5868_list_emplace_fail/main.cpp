#include <list>
#include <cassert>

int main()
{
  std::list<int> l;
  l.emplace_back(5);
  l.emplace_front(6);
  // emplace_front puts the element at the front, not the back.
  assert(l.back() == 6);
  return 0;
}
