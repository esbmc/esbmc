#include <list>
#include <map>
#include <cassert>

struct Inst
{
  int x;
};

// std::list iterators are bidirectional, so they are not LessThanComparable;
// keying a map on one needs a user-supplied operator<, which ADL has to find
// through the element type's namespace. This is the shape goto_programt uses
// for std::map<goto_programt::targett, unsigned>.
bool operator<(
  const std::list<Inst>::iterator a,
  const std::list<Inst>::iterator b)
{
  return a->x < b->x;
}

int main()
{
  std::list<Inst> l;
  Inst i0 = {1};
  l.push_back(i0);

  std::map<std::list<Inst>::iterator, unsigned> m;
  m[l.begin()] = 7;
  assert(m[l.begin()] == 8);
  return 0;
}
