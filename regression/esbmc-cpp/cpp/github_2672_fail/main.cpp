#include <cassert>
class Test
{
public:
  int value;
  int get() { return value; }
  int twice() { return 2 * value; }
};

int main()
{
  Test t;
  t.value = 8;

  int (Test::*memPtr)() = &Test::get;
  memPtr = &Test::twice; // reassignment must be respected
  assert((t.*memPtr)() == 8); // must fail: twice() returns 16
  return 0;
}
