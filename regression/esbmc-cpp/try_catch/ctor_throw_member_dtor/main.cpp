// [except.ctor]/3: a constructor that exits by an exception destroys the
// members it has fully constructed, newest first.
#include <cassert>

int order[4];
int n = 0;

struct M
{
  int v;
  M(int x = 0) : v(x)
  {
    if (x == 9)
      throw 2;
  }
  ~M()
  {
    order[n++] = v;
  }
};

struct Thrower
{
  Thrower()
  {
    throw 1;
  }
};

struct A
{
  M a;
  M b[2];
  M c;
  A(int k) : a(1), b(), c(k)
  {
    if (k == 4)
      throw 3;
  }
};

struct F
{
  M a;
  Thrower t;
  F() : a(5), t()
  {
  }
};

int main()
{
  {
    A x(7);
    assert(n == 0);
  }
  assert(n == 4 && order[0] == 7 && order[3] == 1);

  n = 0;
  try
  {
    A x(9);
  }
  catch (int e)
  {
    assert(e == 2);
  }
  assert(n == 3 && order[0] == 0 && order[2] == 1);

  n = 0;
  try
  {
    A x(4);
  }
  catch (int e)
  {
    assert(e == 3);
  }
  assert(n == 4 && order[0] == 4 && order[3] == 1);

  n = 0;
  try
  {
    F f;
  }
  catch (int e)
  {
    assert(e == 1);
  }
  assert(n == 1 && order[0] == 5);
  return 0;
}
