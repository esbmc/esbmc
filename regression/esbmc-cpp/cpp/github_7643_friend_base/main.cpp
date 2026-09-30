// github.com/esbmc/esbmc/issues/7643: same re-entrancy as
// github_7643_nested_base, reached through the friend-declared class template
// specialization that is the shape the record-definition redirect targets.
#include <cassert>

template <class T>
struct H;

struct A
{
  friend struct H<A>;
  H<A> *p;
  virtual void f()
  {
  }
  int x;
};

template <>
struct H<A> : A
{
  int y;
};

int main()
{
  H<A> h;
  h.x = 1;
  h.y = 2;
  h.p = &h;
  assert(h.p->x == 1);
  return 0;
}
