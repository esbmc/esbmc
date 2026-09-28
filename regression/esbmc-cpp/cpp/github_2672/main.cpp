// Pointer-to-member support (github #2672): .* and ->* on both data and
// function members, including reassignment and passing as arguments.
#include <cassert>

class Test
{
public:
  int value;
  int get() { return value; }
  int twice() { return 2 * value; }
};

int call(Test *p, int (Test::*m)())
{
  return (p->*m)();
}

int main()
{
  Test t;
  t.value = 8;

  int (Test::*memPtr)() = &Test::get;
  assert((t.*memPtr)() == 8);
  memPtr = &Test::twice;
  assert((t.*memPtr)() == 16);

  assert(call(&t, &Test::get) == 8);
  assert(call(&t, &Test::twice) == 16);

  int Test::*vPtr = &Test::value;
  assert(t.*vPtr == 8);
  t.*vPtr = 5;
  assert(t.value == 5);
  Test *tp = &t;
  assert(tp->*vPtr == 5);
  return 0;
}
