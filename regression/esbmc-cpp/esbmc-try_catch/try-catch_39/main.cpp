#include<cassert>

struct Base {};
struct Base1 {};
struct Derived : Base, Base1 {};

int main()
{
  try {
    throw Derived();
  }
  catch(Base1) {  }
  catch(Base) { assert(0); }
  return 0;
}
