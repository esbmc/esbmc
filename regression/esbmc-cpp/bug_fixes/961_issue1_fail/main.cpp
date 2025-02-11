#include <cassert>

class Vehicle
{
public:
  int x;
  Vehicle()
  {
    x = 1;
  }

  int empty()
  {
    return x;
  }
};

int main()
{
  Vehicle xx = Vehicle();
  assert(xx.empty() == 0); // FAIL as it should be 1

  return 0;
}
