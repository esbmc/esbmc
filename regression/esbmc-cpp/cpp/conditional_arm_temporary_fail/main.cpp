// c false never constructs P(1), so dtors stays 0.
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
  assert(dtors == 1);
  return 0;
}
