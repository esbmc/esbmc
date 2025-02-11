#include <cassert>

class Vehicle
{
public:
  int data;
  Vehicle() { data = 1; }
};

class Motorcycle: public Vehicle
{
public:
  Motorcycle() : Vehicle() {}
};

int main()
{
  Motorcycle m;
  bool res1;
  res1 = (m.data == 1); // PASS
  assert(res1);
  return 0;
}

