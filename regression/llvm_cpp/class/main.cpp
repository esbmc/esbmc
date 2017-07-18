#include <assert.h>

int x = 0;

namespace FOO {

class BLAH
{
  public:

    BLAH(float _z, int _y) : z(_z), y(_y) { }
    ~BLAH() { x++; }

    BLAH() { y = 3; };

    int get_A(int a) { int z = 1; return z; }
    int get_A2() { return this->get_A5(); }
    static int get_A5() { return 2; }
    int get_A3() { return this->get_A2(); }
    int get_A1() { return this->y; }
    int get_A4();

    operator bool() { return true; }

    friend BLAH operator+(BLAH lhs, const BLAH& rhs)
    {
      lhs.z += rhs.z;
      return lhs;
    }
    int z;

  protected:

    int y;
    int w;
};

};

int FOO::BLAH::get_A4() { return this->y; }

FOO::BLAH bloh;

int main(int argc, char* argv[])
{
  FOO::BLAH *blah = new FOO::BLAH(1,3);

  {
    FOO::BLAH bleh(1,3);
    FOO::BLAH blih(2,3);
    FOO::BLAH bloh(blih);
    FOO::BLAH bluh = bleh;
    FOO::BLAH(1,3);

    assert(bleh.get_A(4.0f) == 1);
    assert(bleh.get_A1() == 3);
    assert(bleh.get_A3() == 2);

    assert(blih.get_A4() == 3);
    assert(blih.get_A5() == blih.z);

    assert(FOO::BLAH::get_A5() == 2);

    assert(bloh.get_A4() == 3);
    assert(bloh.get_A5() == blih.z);

    assert(bluh.get_A4() == 3);
    assert(bleh.get_A2() == 2);

    assert((bleh + blih).z == 3);
  }

  assert(x == 7);

  return 0;
}


