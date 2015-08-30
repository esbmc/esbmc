// bad_cast example
#include <typeinfo>
#include <cassert>
using namespace std;

class Base {virtual void Member(){}};
class Derived : public Base {};

int main () {
  try
  {
    Base b;
    Derived& rd = dynamic_cast<Derived&>(b);
  }
  catch (bad_cast& bc)
  {
     assert(0);
  }
  return 0;
}
