#include <cassert>

int built = 0, destroyed = 0;

struct C
{
  C()
  {
    if (built == 2)
      throw 1;
    ++built;
  }
  ~C()
  {
    ++destroyed;
  }
};

int main()
{
  try
  {
    C *p = new C[3];
    delete[] p;
  }
  catch (int)
  {
  }
  // The two constructed elements were destroyed when the third threw.
  assert(destroyed == 0);
}
