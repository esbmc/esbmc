#include<cassert>

struct Base {};
struct Base1 {};
struct Base2 {};
struct Derived : Base2, Base, Base1  {};

int main()
{
  try {
    throw Derived();
  }
  catch(Base2) {  }
  catch(Base1) {  }
  catch(Base) { assert(0); }
  return 0;
}
