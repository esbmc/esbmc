#include <cassert>

class Vehicle
{
public:
  int x;
  Vehicle()
  {
    x = 1;
  }

  int do_something()
  {
    return x;
  }
};

int main()
{
  Vehicle xx;
  int y = xx.do_something();
  assert(y == 0); // FAIL as it should be 1

  return 0;
}
