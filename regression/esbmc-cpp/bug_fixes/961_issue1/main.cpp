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
  assert(xx.empty() == 1);

  return 0;
}
