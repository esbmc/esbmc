#include <cassert>

static int destroyed = 0;

struct C
{
  ~C()
  {
    ++destroyed;
  }
};

struct S
{
  union
  {
    char e;
    C c;
  };
  S() : e()
  {
  }
  ~S()
  {
  }
};

union U
{
  char e;
  C c;
  U() : e()
  {
  }
  ~U()
  {
  }
};

int main()
{
  {
    S s;
  }
  assert(destroyed == 0);
  {
    U u;
  }
  assert(destroyed != 0);
  return 0;
}
