// A temporary in an expression statement is destroyed at the end of that
// full-expression ([class.temporary]/4), not at the end of the block.
#include <cassert>

int live = 0;

struct C
{
  int v;
  C(int x) : v(x)
  {
    ++live;
  }
  C(const C &o) : v(o.v)
  {
    ++live;
  }
  ~C()
  {
    --live;
  }
};

int get(const C &c)
{
  assert(live == 1);
  return c.v;
}

void take(C c)
{
  assert(c.v == 2);
}

int main()
{
  int k = 0;
  get(C(1));
  assert(live == 0);
  take(C(2));
  assert(live == 0);
  k = get(C(3));
  assert(live == 0 && k == 3);
  k += get(C(4));
  assert(live == 0 && k == 7);
  take(C(2)), k = 1;
  assert(live == 0);
  for (int i = 0; i < 3; i = i + get(C(1)))
    assert(live == 0);
  assert(live == 0);
  return 0;
}
