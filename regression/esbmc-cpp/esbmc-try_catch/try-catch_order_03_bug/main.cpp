#include<cassert>

struct Base {};
struct Derived : Base {};

int main()
{
  try {
    throw Derived();
  }
  catch(Base) { assert(0); }
  return 0;
}
