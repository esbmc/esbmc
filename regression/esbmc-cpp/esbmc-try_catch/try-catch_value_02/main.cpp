#include <cassert>

class X {
  public:
    X() { throw 5; }
};

int main()
{
  try {
    X x;
  }
  catch(int e) {
    assert(e==5);
  }
  return 0;
}
