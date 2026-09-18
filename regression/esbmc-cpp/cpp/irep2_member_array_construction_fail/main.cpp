#include <cassert>

int marker = 1;

// The per-element calls a class-typed member array expands to carry the whole
// array's type. Folded with that type, the statement reads as array-valued and
// is wrapped in `&stmt[0]`; the call then acquires a temporary whose elements
// are destroyed without ever having been constructed.
struct E
{
  int *p;
  E() : p(&marker)
  {
  }
  ~E()
  {
    assert(p == &marker);
  }
};

struct H
{
  E buf[3];
};

int main()
{
  H h;
  assert(*h.buf[2].p == 2);
}
