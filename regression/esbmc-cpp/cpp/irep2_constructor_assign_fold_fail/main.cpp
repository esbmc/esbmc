#include <cassert>

// The fold's negative half: the assertion is false, so the violated property
// must be main's own. Unfolded, the run fails inside Pool instead.
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
  assert(l.p.buf[0] == 2);
}
