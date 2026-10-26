// A temporary materialized in one arm of ?: or && is destroyed only when that
// arm ran ([class.temporary]/4). Matches g++ -fsanitize=address,undefined
// with c true and false.
#include <cassert>

bool nondet_bool();

int dtors = 0;

struct P
{
  int v;
  P(int x) : v(x)
  {
  }
  ~P()
  {
    ++dtors;
  }
};

int main()
{
  bool c = nondet_bool();

  int x = c ? P(1).v : 0;
  assert(x == (c ? 1 : 0));
  assert(dtors == (c ? 1 : 0));

  dtors = 0;
  int y = c ? P(2).v : P(3).v;
  assert(y == (c ? 2 : 3));
  assert(dtors == 1);

  dtors = 0;
  bool d = c;
  int z = d ? (d = false, P(4).v) : 0;
  assert(z == (c ? 4 : 0));
  assert(dtors == (c ? 1 : 0));

  dtors = 0;
  bool b = c && P(5).v == 5;
  assert(b == c);
  assert(dtors == (c ? 1 : 0));

  return 0;
}
