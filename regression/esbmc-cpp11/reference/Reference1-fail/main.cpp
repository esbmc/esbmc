#include <cassert>

class t1 {
public:
  int i;

  t1(): i(0)
  {
  }
};

int main()
{
  t1 instance1;
  t1 &r = instance1;
  assert(r.i == 0); // pass
  r.i = 1;
  assert(r.i == 2); // fail
}
