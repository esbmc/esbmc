#include <cassert>

class t1 {
public:
  int i;

  t1(): i(0)
  {
  }
};

// Test object reference as function argument
void increment(t1 &t) {
  t.i += 1;
}

int main()
{
  t1 instance1;
  t1 &r = instance1;
  assert(r.i == 0); // pass
  increment(r);
  assert(r.i == 1); // pass
}
