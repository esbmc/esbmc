#include <cassert>

struct A
{
  int a;
  A() : a(1)
  {
  }
};

struct P
{
  virtual ~P()
  {
  }

  int p;

  P() : p(9)
  {
  }
};

struct AP : A, P
{
};

struct PA : P, A
{
};

int main()
{
  try
  {
    throw AP();
  }
  catch (A &a)
  {
    assert(a.a == 1);
  }
  catch (...)
  {
    assert(false);
  }

  try
  {
    throw PA();
  }
  catch (A &a)
  {
    assert(a.a == 1);
  }
  catch (...)
  {
    assert(false);
  }
}
