#include <cassert>

int built = 0, destroyed = 0, last = -1;

struct C
{
  int v;
  C() : v(0)
  {
    if (built == 2)
      throw 1;
    ++built;
  }
  C(int x) : v(x)
  {
    if (x < 0)
      throw 2;
    ++built;
  }
  ~C()
  {
    ++destroyed;
    last = v;
  }
};

int nondet_int();

int main()
{
  // The third default-constructed element throws: the first two are destroyed.
  try
  {
    C *p = new C[3];
    delete[] p;
  }
  catch (int e)
  {
    assert(e == 1);
  }
  assert(built == 2 && destroyed == 2);

  // A listed element throws: the one before it is destroyed.
  built = destroyed = 0;
  try
  {
    C *p = new C[3]{C(5), C(-1)};
    delete[] p;
  }
  catch (int e)
  {
    assert(e == 2);
  }
  assert(built == 1 && destroyed == 1 && last == 5);

  // The filler throws after a nondet number of listed elements; newest first.
  built = destroyed = 0;
  int n = nondet_int();
  __ESBMC_assume(n >= 3 && n <= 4);
  try
  {
    C *p = new C[n]{C(7), C(8)};
    delete[] p;
  }
  catch (int e)
  {
    assert(e == 1);
  }
  assert(built == 2 && destroyed == 2 && last == 7);

  // A two-dimensional array is unwound leaf by leaf.
  built = destroyed = 0;
  try
  {
    C(*q)[2] = new C[2][2];
    delete[] q;
  }
  catch (int)
  {
  }
  assert(built == 2 && destroyed == 2);

  // Nothing throws: elements are destroyed only by delete[].
  built = destroyed = 0;
  C *r = new C[2]{C(1), C(2)};
  assert(destroyed == 0);
  delete[] r;
  assert(destroyed == 2);
}
