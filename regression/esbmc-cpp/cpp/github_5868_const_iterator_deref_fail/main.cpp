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
  // A nondet operator-> would leave this unfalsifiable; it must be refuted.
  assert(it->v == 4);
  return 0;
}
