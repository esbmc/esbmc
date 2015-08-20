#include <cassert>
// PR c++/36089
// { dg-do run }


int f ()
{
  const int c(2);
  int d[c] = { 0, 0 };
  return d[0] + sizeof d;
}

struct A
{
  static int f ()
  {
    const int c(2);
    int d[c] = { 0, 0 };
    return d[0] + sizeof d;
  }
};

template <int> struct B
{
  static int f ()
  {
    const int c = 2;
    int d[c] = { 0, 0 };
    return d[0] + sizeof d;
  }
};

template <int> struct C
{
  static int f ()
  {
    const int c(2);
    int d[c] = { 0, 0 };
    return d[0] + sizeof d;
  }
};

template <int> struct D
{
  static int f ()
  {
    const int e(2);
    const int c(e);
    int d[c] = { 0, 0 };
    return d[0] + sizeof d;
  }
};

int
main (void)
{
  int v = f ();
  if (v != 2 * sizeof (int))
    assert(0);
  if (v != A::f ())
    assert(0);
  if (v != B<6>::f ())
    assert(0);
  if (v != C<0>::f ())
    assert(0);
  if (v != D<1>::f ())
    assert(0);
}

