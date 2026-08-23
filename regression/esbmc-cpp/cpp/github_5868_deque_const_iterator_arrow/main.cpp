#include <deque>
#include <cassert>

struct box
{
  int v;
};

int main()
{
  std::deque<box> d;
  box b;
  b.v = 3;
  d.push_back(b);

  const std::deque<box> &cd = d;
  std::deque<box>::const_iterator it = cd.begin();
  // operator-> was declared and never defined, so this returned a
  // nondeterministic value that satisfied neither the property nor its negation.
  assert(it->v == 3);
  assert((*it).v == 3);
  return 0;
}
