#include <cassert>

// A member whose type has a user-declared default constructor reaches the
// adjuster as `p = Pool()`; without the fold to `Pool(&p)` the constructor
// runs with no object argument.
struct Pool
{
  int buf[4];
  Pool()
  {
    buf[0] = 1;
  }
};

struct L
{
  Pool p;
  L()
  {
    p.buf[1] = 7;
  }
};

int main()
{
  L l;
  assert(l.p.buf[0] == 1 && l.p.buf[1] == 7);
}
