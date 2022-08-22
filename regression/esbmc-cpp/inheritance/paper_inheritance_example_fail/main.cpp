#include <cassert>

class Vehicle
{
public:
  Vehicle() {}
  virtual int number_of_wheels() = 0;
};

class Motorcycle: public Vehicle
{
public:
  Motorcycle() : Vehicle() {}
  virtual int number_of_wheels() { return 2; }
};

class Car: public Vehicle
{
public:
  Car() : Vehicle() {}
  virtual int number_of_wheels() { return 4; }
};

int main()
{
  bool foo = false;

  Vehicle* v;
  if (foo)
    v = new Motorcycle();
  else
    v = new Car();

  bool res;
  res = (v->number_of_wheels() == 2); // should be 4

  assert(res);
  return 0;
}

