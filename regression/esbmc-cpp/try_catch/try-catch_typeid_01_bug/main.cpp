// bad_typeid example
#include <cassert>
#include <typeinfo>
using namespace std;

class Polymorphic {virtual void Member(){}};

int main () {
  try
  {
    Polymorphic * pb = 0;
    const char* name = typeid(*pb).name();
  }
  catch (bad_typeid& bt)
  {
    assert(0);
  }
  return 0;
}
