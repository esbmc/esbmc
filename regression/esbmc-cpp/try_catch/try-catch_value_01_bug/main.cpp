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
    switch(e) {
      case 5:
        assert(0);
        break;

      default:
        return 1;
    }
  }
  return 0;
}
